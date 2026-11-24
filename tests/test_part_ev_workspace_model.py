import copy
import unittest
import torch
from torch.nn import functional as F
from part_ev_workspace.attention import Attention
from part_ev_workspace.model import build_model, WorkspaceStage, GlobalWorkspace

torch.set_num_threads(1)


def inputs(n=7,b=3):
    g=torch.Generator().manual_seed(73)
    vectors=torch.randn(b,4,n,generator=g)
    vectors[:,3]=vectors[:,:3].square().sum(1).sqrt()+1
    valid=torch.arange(n)[None]<torch.tensor([max(1,n-i) for i in range(b)])[:,None]
    return [torch.zeros(b,2,n),torch.randn(b,17,n,generator=g),vectors,valid[:,None]]


def active(module):
    with torch.no_grad():
        for name,p in module.named_parameters():
            if name.endswith(('write.output.weight','class_read.attention.output.weight')):
                p.normal_(std=.01)


class WorkspaceMathTests(unittest.TestCase):
    def test_attention_independent_double_oracle_and_gradients(self):
        torch.manual_seed(9)
        a=Attention().double()
        q=torch.randn(2,3,128,dtype=torch.float64,requires_grad=True)
        s=torch.randn(2,5,128,dtype=torch.float64,requires_grad=True)
        valid=torch.tensor([[1,1,0,1,0],[0,1,0,0,0]],dtype=torch.bool)
        actual,health=a(q,s,valid)
        records=[]
        for b in range(2):
            queries=F.linear(q[b],a.query.weight,a.query.bias)
            keys=F.linear(s[b,valid[b]],a.key.weight,a.key.bias)
            values=F.linear(s[b,valid[b]],a.value.weight,a.value.bias)
            heads=[]
            for h in range(8):
                region=slice(h*16,(h+1)*16)
                probs=(queries[:,region]@keys[:,region].T/4).softmax(-1)
                heads.append(probs@values[:,region])
            records.append(F.linear(torch.cat(heads,-1),a.output.weight,a.output.bias))
        expected=torch.stack(records)
        torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-10)
        parameters=(q,s,*a.parameters())
        ga=torch.autograd.grad(actual.square().sum(),parameters,retain_graph=True)
        ge=torch.autograd.grad(expected.square().sum(),parameters)
        for x,y in zip(ga,ge): torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-10)
        self.assertTrue(health)

    def test_mask_health_and_empty_attention(self):
        a=Attention()
        q,s=torch.randn(2,3,128),torch.randn(2,5,128)
        valid=torch.ones(2,5,dtype=torch.bool);valid[:,2]=False
        expected,_=a(q,s,valid)
        s[:,2]=float('nan')
        actual,ok=a(q,s,valid)
        torch.testing.assert_close(actual,expected);self.assertTrue(ok)
        _,ok=a(q,s,torch.zeros_like(valid));self.assertFalse(ok)
        s[:,0]=float('nan');_,ok=a(q,s,valid);self.assertFalse(ok)

    def test_stage_mask_permutation_isolation_and_double_gradcheck(self):
        torch.manual_seed(4)
        stage=WorkspaceStage().double();active(stage)
        h=torch.randn(2,4,128,dtype=torch.float64,requires_grad=True)
        z=torch.randn(2,8,128,dtype=torch.float64,requires_grad=True)
        mask=torch.tensor([[1,0,1,1],[1,1,0,1]],dtype=torch.bool)
        out,w,ok=stage(h,z,mask)
        perm=torch.tensor([3,1,0,2]);hp,wp,_=stage(h[:,perm],z,mask[:,perm])
        torch.testing.assert_close(hp,out[:,perm],rtol=1e-9,atol=1e-10)
        torch.testing.assert_close(wp,w,rtol=1e-9,atol=1e-10)
        h2=h.detach().clone();h2[~mask]=1e5
        out2,w2,_=stage(h2,z,mask)
        torch.testing.assert_close(w,w2,rtol=1e-9,atol=1e-10)
        torch.testing.assert_close(out[mask],out2[mask],rtol=1e-9,atol=1e-10)
        one,z1,_=stage(h[:1],z[:1],mask[:1])
        torch.testing.assert_close(one,out[:1]);torch.testing.assert_close(z1,w[:1])
        grads=torch.autograd.grad(out.square().sum()+w.square().sum(),(h,z,*stage.parameters()))
        self.assertTrue(ok);self.assertTrue(all(torch.isfinite(g).all() for g in grads))


class WorkspaceModelTests(unittest.TestCase):
    def test_inventory_independent_stages_and_identity_rng(self):
        torch.manual_seed(101);base=build_model('control',trim=False)
        state_after_base=torch.get_rng_state()
        torch.manual_seed(101);model=build_model(trim=False)
        self.assertTrue(torch.equal(state_after_base,torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in model.parameters()),3720614)
        self.assertEqual(sum(p.numel() for p in model.mod.global_workspace.parameters()),1393152)
        for k,v in base.state_dict().items(): self.assertTrue(torch.equal(v,model.state_dict()[k]),k)
        pointers=[s.read.query.weight.data_ptr() for s in model.mod.global_workspace.stages]
        self.assertEqual(len(set(pointers)),4)
        base.train();model.train();values=inputs()
        torch.manual_seed(99);expected=base(*values);rng=torch.get_rng_state()
        torch.manual_seed(99);actual=model(*values)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(actual.shape,(3,10))

    def test_checkpoint_active_backward_cleanup_and_successive_forwards(self):
        torch.manual_seed(12);fast=build_model(trim=False);active(fast)
        slow=copy.deepcopy(fast);slow.mod.workspace_checkpointed=False
        for model in (fast,slow):
            model.train();torch.manual_seed(77)
            logits=model(*inputs())
            self.assertIsNone(model._state_pair_cache.stem)
            logits.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        for (n,p),(m,q) in zip(fast.named_parameters(),slow.named_parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-5,msg=n)
        fast.eval()
        with torch.no_grad():
            first=fast(*inputs());fast(*inputs(n=4));again=fast(*inputs())
        torch.testing.assert_close(first,again,rtol=1e-5,atol=1e-6)
        self.assertFalse(any('last_workspace' in n for n in fast.state_dict()))

    def test_learning_real_ranger_and_one_particle(self):
        from part_ev_fullscale.optimizer import Ranger
        torch.manual_seed(101);model=build_model(trim=False)
        opt=Ranger(model.parameters());values=inputs(n=5,b=4)
        initial={n:p.detach().clone() for n,p in model.named_parameters() if 'global_workspace' in n}
        for step in range(8):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(*values),torch.arange(4)).backward();opt.step()
        ws=model.mod.global_workspace
        for i,stage in enumerate(ws.stages):
            self.assertGreater(float(stage.write.output.weight.abs().max()),0)
            key=f'mod.global_workspace.stages.{i}.read.query.weight'
            self.assertFalse(torch.equal(initial[key],dict(model.named_parameters())[key]))
        opt.zero_grad(set_to_none=True);model(*inputs(n=1,b=3)).sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=='__main__': unittest.main()
