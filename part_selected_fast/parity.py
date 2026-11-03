"""Reference/candidate admission without mutating a production stream or state."""
import copy
import io
from contextlib import nullcontext
import numpy as np
import torch
from part_ev_fullscale.checkpoint import capture_rank_state, restore_rank_state
from part_ev_screen.precision import model_precision
from part_selected_screen.memory import compare, PARITY
from .trees import decode, oracle_batch


POLICY = dict(version='selected_fast_parity_v1', numerical=PARITY,
              exact_tree=True, modes=['fp32_eval','fp32_train','fp16_train'],
              trimming_checks=8, optimizer_updates=6, lr_boundary_steps=[139999,140000],
              serialized_restart_after=3, family_features=True)


def compare_states(actual, expected, limits):
    if set(actual) != set(expected):
        raise ValueError('State keys differ')
    for key in actual:
        a,b = actual[key], expected[key]
        if a.is_floating_point():
            compare(a,b,limits)
        elif not torch.equal(a,b):
            raise ValueError(f'Discrete state differs: {key}')


def tree_parity(provider, reference_backend, values):
    from part_selected_screen.data import raw_tokens
    _,f,v,m = values
    raw = raw_tokens(f,v,m)
    expected = oracle_batch(v.float().transpose(1,2).cpu().numpy(),raw.float().cpu().numpy(),
                            m[:,0].bool().cpu().numpy(),reference_backend)
    expected = decode(*expected, v.shape[-1], v.device)
    actual = provider(v,raw,m)
    if set(actual) != set(expected):
        raise ValueError('Tree packed fields differ')
    for key in actual:
        if not torch.equal(actual[key], expected[key]):
            raise ValueError(f'Batched tree mismatch: {key}')
    return dict(passed=True, fields=len(actual), exact=True)


def execution_parity(candidate, reference, batch, device):
    """Run train/eval/AMP and native-trim checks; fully restore both models."""
    state = copy.deepcopy(candidate.state_dict())
    reference_state = copy.deepcopy(reference.state_dict())
    if set(state) != set(reference_state):
        raise ValueError('Parameter/buffer keys differ')
    rng = capture_rank_state(candidate)
    reference_rng = capture_rank_state(reference)
    models = (reference,candidate)
    modes = [[(m,m.training) for m in model.modules()] for model in models]
    trimmers = [(m,m.enabled) for model in models for m in model.modules() if type(m).__name__=='SequenceTrimmer']
    n = min(8,len(batch[1]['_label_']))
    if n < 2:
        raise ValueError('Parity needs at least two jets')
    values = [batch[0][k][:n].to(device) for k in ('pf_points','pf_features','pf_vectors','pf_mask')]
    truth = batch[1]['_label_'][:n].long().to(device)
    result = dict(policy=copy.deepcopy(POLICY), modes={})
    try:
        result['trees'] = tree_parity(candidate.fast_provider, reference.mod.pair_embed.relations.backend, values)
        for m,_ in trimmers:
            m.enabled=False
        for name,training,amp in (('fp32_eval',False,False),('fp32_train',True,False),('fp16_train',True,True)):
            observations=[]
            for model in models:
                model.load_state_dict(state,strict=True); model.zero_grad(set_to_none=True)
                restore_rank_state(model,rng); model.train(training)
                cpu_amp = torch.autocast('cpu',dtype=torch.float16) if amp and device.type=='cpu' else nullcontext()
                features={}; hooks=[]
                relations=model.mod.pair_embed.relations
                def capture(label):
                    def hook(module,args,output):
                        features[label]=output.detach().float().cpu().clone()
                    return hook
                for label,module in relations.normalizers.items():
                    hooks.append(module.register_forward_hook(capture('normalized_'+label)))
                for label in ('pt','track_node','track_pair','region'):
                    hooks.append(getattr(relations,label).register_forward_hook(capture(label)))
                try:
                    with torch.set_grad_enabled(training), model_precision(model,device,training=amp), cpu_amp:
                        logits=model(*values)
                        loss=torch.nn.functional.cross_entropy(logits.float(),truth)
                finally:
                    for hook in hooks: hook.remove()
                if training:
                    loss.backward()
                grads={}
                if training:
                    for k,p in model.named_parameters():
                        if p.grad is None:
                            raise ValueError(f'Missing gradient: {k}')
                        grads[k]=p.grad.detach().float().cpu().clone()
                observations.append((logits.detach().float().cpu(),grads,copy.deepcopy(model.state_dict()),features,
                                     loss.detach().float().cpu()))
            a,b=observations
            limits=PARITY['fp16' if amp else 'fp32']
            row=dict(logits=compare(b[0],a[0],limits),loss=compare(b[4],a[4],limits),gradients={},features={})
            if set(a[3])!=set(b[3]) or len(a[3])!=8: raise ValueError('Missing family features')
            for label in a[3]: row['features'][label]=compare(b[3][label],a[3][label],limits)
            groups={}
            for k in a[1]:
                groups.setdefault(k.rpartition('.')[0],[]).append(k)
            for group,keys in groups.items():
                row['gradients'][group]=compare(torch.cat([b[1][k].flatten() for k in keys]),
                    torch.cat([a[1][k].flatten() for k in keys]),limits,gradients=True)
            compare_states(b[2],a[2],limits)
            result['modes'][name]=row
        # Exercise native RNG/permutation/truncation, including warm-up counter.
        streams=[]
        for model in models:
            model.load_state_dict(state); restore_rank_state(model,rng); model.train()
            model.mod.trimmer.enabled=True
            rows=[]
            with torch.no_grad(), model_precision(model,device,training=False):
                for _ in range(POLICY['trimming_checks']):
                    rows.append(model(*values).float().cpu())
            streams.append((rows,capture_rank_state(model)))
        for a,b in zip(streams[0][0],streams[1][0]):
            compare(b,a,PARITY['fp32'])
        a,b=streams[0][1],streams[1][1]
        if a['python'] != b['python'] or a['trimmers'] != b['trimmers'] or not torch.equal(a['torch'],b['torch']):
            raise ValueError('Native trimming RNG diverged')
        if len(a['cuda'])!=len(b['cuda']) or any(not torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda'])):
            raise ValueError('CUDA RNG diverged')
        if (a['numpy'][0]!=b['numpy'][0] or not np.array_equal(a['numpy'][1],b['numpy'][1])
                or a['numpy'][2:]!=b['numpy'][2:]):
            raise ValueError('NumPy RNG diverged')
        # A finite optimizer pilot including the sixth-step Lookahead update.
        from part_ev_fullscale.optimizer import build_optimizer
        from part_ev_screen.protocol import recipes
        from scripts.run_part_ev_screen import training_step
        small=({k:batch[0][k][:n] for k in batch[0]}, {'_label_':batch[1]['_label_'][:n]}, {})
        pilots=[]
        for model,restart in ((reference,False),(candidate,False),(candidate,True)):
            model.load_state_dict(state); restore_rank_state(model,rng)
            optimizer=build_optimizer(model,recipes()['ranger_short_v1'])
            scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda')
            skips=[]
            for position,step in enumerate([*range(POLICY['optimizer_updates']),*POLICY['lr_boundary_steps']]):
                skips.append(training_step(model,optimizer,scaler,small,device,recipes()['ranger_short_v1'],step)['amp_skipped'])
                if restart and position+1==POLICY['serialized_restart_after']:
                    # Our own freshly created bytes, not an untrusted checkpoint.
                    stream=io.BytesIO()
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                                    scaler=scaler.state_dict(),rng=capture_rank_state(model)),stream)
                    stream.seek(0)
                    saved=torch.load(stream,map_location=device,weights_only=False)
                    model.load_state_dict(saved['model'])
                    optimizer=build_optimizer(model,recipes()['ranger_short_v1'])
                    optimizer.load_state_dict(saved['optimizer'])
                    scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda')
                    scaler.load_state_dict(saved['scaler']); restore_rank_state(model,saved['rng'])
                    if model.fast_provider.cache is not None: model.fast_provider.cache.clear()
            pilots.append((copy.deepcopy(model.state_dict()),optimizer.state_dict(),scaler.state_dict(),skips))
        limits=PARITY['fp16' if device.type=='cuda' else 'fp32']
        compare_states(pilots[1][0],pilots[0][0],limits)
        def walk(a,b):
            if isinstance(a,torch.Tensor):
                if a.is_floating_point(): compare(b,a,limits,gradients=True)
                elif not torch.equal(a,b): raise ValueError('Optimizer discrete tensor differs')
            elif isinstance(a,dict):
                if set(a)!=set(b): raise ValueError('Optimizer/scaler fields differ')
                for k in a: walk(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                if len(a)!=len(b): raise ValueError('Optimizer/scaler length differs')
                for x,y in zip(a,b): walk(x,y)
            elif a!=b: raise ValueError('Optimizer/scaler/skip state differs')
        for i in (1,2,3): walk(pilots[0][i],pilots[1][i])
        compare_states(pilots[2][0],pilots[1][0],limits)
        for i in (1,2,3): walk(pilots[1][i],pilots[2][i])
        result.update(passed=True,optimizer_updates=POLICY['optimizer_updates'],native_trimming=True,
                      serialized_restart=True,lr_boundary_steps=POLICY['lr_boundary_steps'])
        return result
    finally:
        for model,saved,model_modes in zip(models,(reference_state,state),modes):
            model.load_state_dict(saved); model.zero_grad(set_to_none=True)
            model.mod.pair_embed.clear()
            for module,training in model_modes: module.training=training
        for module,enabled in trimmers: module.enabled=enabled
        restore_rank_state(reference,reference_rng); restore_rank_state(candidate,rng)
        if candidate.fast_provider.cache is not None: candidate.fast_provider.cache.clear()
