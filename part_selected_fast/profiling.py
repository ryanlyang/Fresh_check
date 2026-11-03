"""Bounded measurement helpers; no trace/cache persistence unless requested."""
import os
from pathlib import Path
import threading
import time
import types
import torch


def host_memory_limit():
    """Smallest visible physical/cgroup/Slurm allowance; no resource changes."""
    limits=[]
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemTotal:'): limits.append(int(line.split()[1])*1024)
        if os.environ.get('SLURM_MEM_PER_NODE'):
            limits.append(int(os.environ['SLURM_MEM_PER_NODE'])*1024**2)
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            _,controllers,name=line.split(':',2)
            if not controllers:
                parent=Path('/sys/fs/cgroup')/name.lstrip('/')
                stop=Path('/sys/fs/cgroup')
                files=('memory.max',)
            elif 'memory' in controllers.split(','):
                parent=Path('/sys/fs/cgroup/memory')/name.lstrip('/')
                stop=Path('/sys/fs/cgroup/memory'); files=('memory.limit_in_bytes',)
            else: continue
            while parent.is_relative_to(stop):
                for filename in files:
                    try:
                        value=(parent/filename).read_text().strip()
                        if value.isdecimal() and int(value)>0: limits.append(int(value))
                    except OSError: pass
                if parent==stop: break
                parent=parent.parent
    except (OSError,ValueError): pass
    return min(limits) if limits else None


def rss_tree(pid=None):
    """Linux process + live descendants; vanished processes are harmless."""
    pid=os.getpid() if pid is None else pid
    seen=set()
    def visit(pid):
        if pid in seen: return 0
        seen.add(pid)
        try:
            pages=int(Path(f'/proc/{pid}/statm').read_text().split()[1])
            children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        except (OSError,ValueError,IndexError): return 0
        return pages*os.sysconf('SC_PAGE_SIZE')+sum(visit(int(child)) for child in children)
    return visit(pid)


class MemoryMonitor:
    def __init__(self):
        self.peak=0
        self.gpu_reserved=0
        self.stop=threading.Event()
    def __enter__(self):
        def sample():
            while not self.stop.is_set():
                self.peak=max(self.peak,rss_tree())
                if torch.cuda.is_initialized():
                    self.gpu_reserved=max(self.gpu_reserved,torch.cuda.max_memory_reserved())
                self.stop.wait(.2)
        self.worker=threading.Thread(target=sample,daemon=True)
        self.worker.start()
        return self
    def __exit__(self,*_):
        self.peak=max(self.peak,rss_tree())
        self.stop.set(); self.worker.join()


def instrument(model):
    """Label pure feature boundaries on this model only. Return a restore hook."""
    relation=model.mod.pair_embed.relations
    old=relation.forward
    fn=old.__func__
    namespace=dict(fn.__globals__)
    def wrap(name,func):
        def wrapped(*a,**kw):
            with torch.profiler.record_function('selected_profile/'+name): return func(*a,**kw)
        return wrapped
    for name in ('build_pt_raw_features','build_track_compatibility','trees_for','region_features'):
        namespace[name]=wrap(name,namespace[name])
    object.__setattr__(relation,'forward',types.MethodType(types.FunctionType(fn.__code__,namespace,fn.__name__),relation))
    handles=[]
    for name,module in model.named_modules():
        if name in ('mod.trimmer','mod.pair_embed','mod.blocks','mod.cls_blocks') or name.startswith('mod.pair_embed.relations.') and name.count('.')==3:
            contexts=[]
            def pre(module,args,*,label=name,contexts=contexts):
                scope=torch.profiler.record_function('selected_profile/'+label); scope.__enter__(); contexts.append(scope)
            def post(module,args,result,*,contexts=contexts):
                if contexts: contexts.pop().__exit__(None,None,None)
            handles += [module.register_forward_pre_hook(pre),module.register_forward_hook(post,always_call=True)]
    def restore():
        object.__setattr__(relation,'forward',old)
        for handle in handles: handle.remove()
    return restore


def trace_steps(model,optimizer,scaler,batches,device,recipe,training_step,trace_path):
    restore=instrument(model)
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True,profile_memory=True) as profile:
            for index,batch in enumerate(batches):
                with torch.profiler.record_function('selected_profile/train_step'):
                    training_step(model,optimizer,scaler,batch,device,recipe,index)
                profile.step()
        profile.export_chrome_trace(str(trace_path))
        rows=[]
        for event in profile.key_averages():
            if event.key.startswith(('selected_profile/','selected_fast/')):
                rows.append(dict(name=event.key,calls=event.count,cpu_total_us=event.cpu_time_total,
                                 device_total_us=getattr(event,'device_time_total',getattr(event,'cuda_time_total',0))))
        return rows
    finally: restore()
