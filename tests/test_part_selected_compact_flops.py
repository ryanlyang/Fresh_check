import pytest
import torch
from part_selected_compact import flops as F, protocol as P
from part_ev_fullscale.protocol import learning_rate
from tests.test_part_selected_compact_model import factory
from tests.test_part_fullscale_model import bounded_cpu_threads


@pytest.mark.parametrize('arch', P.COUNTS)
def test_real_selected_flops_curves_and_unseen_lengths(arch):
    model = factory(arch)
    row = F.curve(model, torch.device('cpu'))
    assert row['directed_pairs'] and row['selected_encoders_included']
    assert row['checkpoint_recomputation_included'] and not row['cpu_tree_work_included']
    for n in (2, 9, 33, 66):
        actual = F.measure(model, n, torch.device('cpu'), backward=True)['per_jet']
        assert actual == F.mean_cost(row['coefficients']['training'], {str(n): 1}, singleton_cost=1)
    # Selected dense encoders must not disappear from the count.
    assert row['samples']['128']['forward']['per_jet'] > (942738944 if arch == 'compact' else 699928064)


def test_fifty_checks_and_stretched_schedule():
    fixed = P.recipe('compact')
    extended = P.recipe('matched', 1300000)
    assert fixed['epochs'] == extended['epochs'] == 50
    assert fixed['pair_bias'] == 'layerwise' and extended['pair_bias'] == 'shared'
    assert fixed['selected_relations'] == extended['selected_relations'] == ['PT', 'TRACK', 'REGION']
    for i in range(50):
        assert learning_rate(i*20000, fixed) == learning_rate(i*26000, extended)
    with pytest.raises(ValueError): P.recipe('compact', 1300000)
    with pytest.raises(ValueError): P.recipe('matched', 1300001)
    row = F.budget(13, 10)
    assert row['matched_steps'] == 1300000 and row['matched_steps_per_check'] == 26000
    assert row['matched_training_flops'] <= row['compact_training_flops']
