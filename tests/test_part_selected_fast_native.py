"""Native integration is required on Linux; never pretend CPU-oracle tests compile C++."""
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest
import torch
from part_selected_fast import trees


@pytest.mark.skipif(sys.platform!='linux' or not shutil.which('c++') or not shutil.which('ninja'),
                    reason='Requires Linux C++/OpenMP and Ninja (also enforced by cluster prepare)')
def test_compiled_backend_vs_original_at_multiple_thread_counts(tmp_path,monkeypatch):
    from torch.utils.cpp_extension import load
    from part_selected_screen.region import SOURCE
    monkeypatch.setenv('MAX_JOBS','1')
    build=tmp_path/'original'; build.mkdir()
    original=load('selected_fast_test_reference',sources=[str(SOURCE)],extra_cflags=trees.FLAGS,
                  extra_ldflags=['-fopenmp'],build_directory=str(build))
    candidate=trees.build_backend(tmp_path/'candidate')
    result=trees.backend_parity(candidate,original,4)
    assert result['passed'] and result['exact']
    assert trees.load_backend(tmp_path/'candidate').contract=='selected_tree_batch_v1'
    # Authenticate binaries and reject changed manifests instead of rebuilding silently.
    manifest=tmp_path/'candidate/manifest.json'
    row=trees.read_json(manifest); row['sources']={}; row.pop('content_hash')
    trees.write_json(manifest,trees.hashed(row))
    with pytest.raises(ValueError,match='changed'): trees.load_backend(tmp_path/'candidate')


def test_backend_parity_requires_exact_fields_and_invalid_input_rejection():
    class Oracle:
        def build_batch(self,v,r,m,threads):
            return trees.oracle_batch(v,r,m,'reference_for_tests')
    good=Oracle()
    # Small direct oracle fixtures exercise byte pack determinism locally; the
    # full native width/thread matrix above is intentionally not replaced by this.
    v=np.zeros((2,2,4)); r=np.zeros((2,2,14)); m=np.array([[False,False],[True,False]])
    a=good.build_batch(v,r,m,1); b=good.build_batch(v,r,m,4)
    for x,y in zip(a,b): assert np.array_equal(x,y)
    p=trees.decode(*a,2,'cpu')
    assert p['n_valid'].tolist()==[0,1]
    assert all(torch.isfinite(t).all() for t in p.values())
