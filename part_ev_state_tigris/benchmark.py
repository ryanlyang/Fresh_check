"""Private ARM adapter of the existing all-eight State-EV benchmark."""
import importlib.util
import os
import platform
import signal
import time

from . import protocol as P


def create_driver():
    module_spec = importlib.util.spec_from_file_location(
        '_tigris_state_probe_adapter', P.ROOT/'scripts/run_part_ev_state_compare.py')
    adapter = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(adapter)
    engine, probe = adapter.D, adapter.probe
    engine.P = adapter.P = probe.P = P

    def gpu_setup(spec):
        torch = engine.torch
        if platform.machine() != 'aarch64' or not torch.cuda.is_available():
            raise RuntimeError('Run on a Tigris ARM CUDA compute node')
        if int(os.environ.get('WORLD_SIZE','1')) != 1 or torch.cuda.device_count() != 1:
            raise RuntimeError('Request exactly one visible GPU')
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        for name in ('SIGTERM','SIGUSR1'):
            signal.signal(getattr(signal,name), engine.stop_signal)
        runtime = dict(gpu=torch.cuda.get_device_name(0),
                       gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                       world_size=1,batch_size=512,workers=2,batchnorm='native',tf32=False,
                       cudnn_benchmark=False,precision=engine.PRECISION_POLICY,
                       campaign_sha256=spec['content_hash'])
        P.validate_runtime(runtime)
        return torch.device('cuda:0'), runtime

    engine.gpu_setup = probe.gpu_setup = gpu_setup
    return engine, probe


def run(args):
    engine, probe = create_driver()
    deadline = time.monotonic()+P.TRANSFER['soft_seconds']
    original_stop = engine.check_stop

    def check_stop():
        original_stop()
        if time.monotonic() >= deadline:
            raise TimeoutError('Two-hour diagnostic budget exhausted; no production training was started')

    engine.check_stop = check_stop
    # The inherited probe constructs disposable model/optimizer state only.
    probe.benchmark(args)
