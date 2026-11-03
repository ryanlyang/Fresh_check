"""Isolated engine namespace: unchanged training/selection/evaluation algorithms."""
import functools
import importlib.util
import types
from pathlib import Path
from part_selected_screen import data
from part_selected_screen.region import load_backend as reference_backend
from part_selected_tigris import engine as original
from scripts.run_part_selected_screen import initial_parity, diagnostics
from part_selected_screen.model import build_model as reference_model
from . import protocol as P
from .trees import load_backend, PackedTrees
from .cache import TreeCache
from .model import build_model
from .parity import execution_parity
from part_ev_fullscale.checkpoint import capture_rank_state, restore_rank_state


def host_peak_bytes():
    import resource
    # Linux ru_maxrss is KiB; add children conservatively (terminated children
    # plus the parent's lifetime peak). Live worker RSS is measured separately.
    own=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    children=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024
    return int(own+children)


def gpu_setup(self,spec):
    from .profiling import host_memory_limit
    requested=int(spec['launch']['memory_gib'])*2**30
    actual=host_memory_limit()
    if actual is not None and requested>actual:
        raise RuntimeError('Job/cgroup host memory allocation is below the frozen request')
    device,runtime=original.gpu_setup(self,spec)
    runtime['execution']=spec['execution']
    return device,runtime


def engine_for(output):
    output=Path(output)
    spec=P.load_campaign(output)
    import part_screen_common.engine as common
    module_spec=importlib.util.spec_from_file_location('part_screen_common._selected_fast_engine',common.__file__)
    module=importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    module.build_loader=functools.partial(data.build_loader,config=output/'inputs/selected.yaml')
    module.finite_validation_loader=functools.partial(data.finite_validation_loader,config=output/'inputs/selected.yaml')
    module.batch_fingerprint=data.batch_fingerprint
    normal=P.checked_json(output/'inputs/normalization.json','part_selected_normalization_v1')
    old_backend=reference_backend(Path(spec['reference']['path'])/'backend')
    native=load_backend(output/'backend')
    options=spec['execution']
    def factory(arch,**kwargs):
        cache=TreeCache(options['cache_bytes']) if options['cache_bytes'] else None
        provider=PackedTrees(native,threads=options['tree_threads'],cache=cache,namespace=spec['content_hash'])
        return build_model(arch,normalization=normal,backend=old_backend,provider=provider,
                           cache_validation=cache is not None,compile_region=options['compile_region'],**kwargs)
    def parity(model,task,batch,device):
        rng=capture_rank_state(model)
        try:
            result=initial_parity(model,task,batch,device)
            reference=reference_model(task['architecture'],normalization=normal,backend=old_backend,
                                      seed=task['seed'],trim=True,use_amp=False).to(device)
            result['fast_execution']=execution_parity(model,reference,batch,device)
            return result
        finally: restore_rank_state(model,rng)
    class Engine(module.ScreenEngine):
        gpu_setup=gpu_setup
        # Its protocol references are all through self.P (unlike gpu_setup).
        lock=original.lock
        def benchmark(self,args):
            path=args.output/'benchmarks'/f"{P.TASKS[args.task_index]['run_id']}.json"
            # Call the unchanged sustained benchmark, then augment only the new
            # campaign's own record. Production gate requires this augmentation.
            from .profiling import MemoryMonitor
            import torch
            with MemoryMonitor() as memory:
                original.benchmark(self,args)
            row=P.read_json(path)
            row['host_peak_bytes']=max(row.get('host_peak_bytes',0),memory.peak,host_peak_bytes())
            # max_memory_reserved is reset by the old benchmark between phases;
            # sample reserved bytes throughout instead of mistaking its last
            # validation phase for the whole-run maximum.
            row['gpu_peak_reserved_bytes']=max(row.get('gpu_peak_reserved_bytes',0),memory.gpu_reserved,
                                               torch.cuda.max_memory_reserved())
            profile=P.checked_json(args.output/'profiles'/f"{P.TASKS[args.task_index]['run_id']}.json",'selected_fast_profile_v1')
            startup=max(r['end_to_end_seconds']-r['seconds'] for r in profile['measurements']['candidate'])
            if 'startup_allowance_seconds' not in row:
                row['projected_training_hours']+=startup/3600
                row['startup_allowance_seconds']=startup
            P.write_json(path,row)
    return Engine(P,factory,parity,diagnostics)
