import copy
import pytest
from part_ev_state_speed import protocol as P
from part_ev_state_speed.benchmark import summarize_samples
from part_ev_message_study import protocol as old
from part_ev_screen import protocol as screen
from tests.test_part_ev_message_protocol import parity_fixture
from tests.test_part_ev_screen_protocol import precision_record


def hashed(row):
    row["content_hash"] = P.digest({k:v for k,v in row.items() if k != "content_hash"})
    return row


def fixture(root):
    runtime = dict(gpu="A100", gpu_memory_bytes=1000)
    spec = dict(content_hash="campaign", donor=dict(path=str(root/"donor"), runtime=runtime),
                state_reference=dict(checkpoint_sha256="trained"))
    P.write_json(root/"donor/benchmarks/EV_RANGER.json", dict(input_fingerprints={"0":"batch"}))
    for task in P.TASKS:
        parity = parity_fixture()
        parity["state_speed_equivalence"] = dict(passed=True, nonzero_branch=True, policy=P.SPEED_POLICY,
            gradients=dict(passed=True, checkpoint_backward=True, tensors=19, forward=dict(passed=True)),
            trained_checkpoint=dict(passed=True, checkpoint_sha256="trained", events=64))
        row = dict(task=task, campaign_sha256="campaign", runtime=dict(runtime, campaign_sha256="campaign"),
            passed=True, measured_steps=4200, amp_skipped_steps=2, initial_parity=parity,
            training_precision=precision_record("torch.float16"),
            validation=dict(precision=precision_record("torch.float32"), events=128000, seconds=100),
            input_fingerprints={"0":"batch"}, native_peak_bytes=100, full_length_peak_bytes=200,
            validation_peak_bytes=300, projected_training_hours=10,
            full_length_attempts=[dict(attempt=i+1, loss=1., state_finite=True, amp_skipped=False,
                                      gradient_norm=2.) for i in range(18)])
        P.write_json(root/"benchmarks"/(task["run_id"]+".json"), hashed(row))
        speed = dict(passed=True, task=task, campaign_sha256="campaign", policy=P.SPEED_POLICY,
            device_type="cuda", runtime=runtime, particle_slots=128, valid_particles=40000,
            weights="fresh_seeded_weights_for_speed_only_not_accuracy", input_fingerprint="batch",
            models={name: {str(b):dict(summarize_samples([.1,.2,.3], b), peak_bytes=300)
                           for b in (1,512)} for name in ("ParT","EV","STATE_original",task["run_id"])})
        P.write_json(root/"benchmarks"/(task["run_id"]+"_speed.json"), hashed(speed))
    return spec


def test_private_protocol_keeps_historical_contracts_intact():
    assert len(old.TASKS) == 5 and len(screen.TASKS) == 7 and len(P.TASKS) == 3
    assert old.SCHEMA == "part_ev_message_study_v1"
    assert P._review_core.__globals__ is not old.benchmark_summary.__globals__
    assert P.recipes() == old.recipes()
    assert {t["seed"] for t in P.TASKS} == {101}
    for bad in (-1, 3, None, True):
        with pytest.raises(ValueError):
            P.task_directory("/example", bad)
    for step in (0, 139999, 140000, 196000, 199999):
        assert P.learning_rate(step, P.recipes()["ranger_short_v1"]) == old.learning_rate(step, old.recipes()["ranger_short_v1"])


@pytest.mark.parametrize("failure", ["missing", "failed", "policy", "device", "runtime", "model", "batch",
    "samples", "negative_time", "summary", "peak", "input", "slots", "weights", "parity", "zero_branch",
    "gradients", "backward", "trained_hash", "events", "benchmark_hash"])
def test_speed_and_nonzero_equivalence_gates(tmp_path, failure):
    spec = fixture(tmp_path)
    assert P.benchmark_summary(tmp_path,spec)["recommended_train_hours"] == 13
    training = failure in {"parity","zero_branch","gradients","backward","trained_hash","events","benchmark_hash"}
    path = tmp_path/"benchmarks"/(P.TASKS[0]["run_id"] + (".json" if training else "_speed.json"))
    row = P.read_json(path)
    if failure == "missing":
        path.unlink()
    else:
        if failure == "failed": row["passed"] = False
        if failure == "policy": row["policy"]["repeats"] = 1
        if failure == "device": row["device_type"] = "cpu"
        if failure == "runtime": row["runtime"]["gpu"] = "H100"
        if failure == "model": row["models"].pop("EV")
        if failure == "batch": row["models"]["EV"].pop("1")
        if failure == "samples": row["models"]["EV"]["512"]["samples_seconds"].pop()
        if failure == "negative_time": row["models"]["EV"]["512"]["samples_seconds"][0] = -1
        if failure == "summary": row["models"]["EV"]["512"]["jets_per_second"] = 9999
        if failure == "peak": row["models"]["EV"]["512"]["peak_bytes"] = 999
        if failure == "input": row["input_fingerprint"] = "wrong"
        if failure == "slots": row["particle_slots"] = 32
        if failure == "weights": row["weights"] = "unknown"
        if training:
            evidence = row["initial_parity"]["state_speed_equivalence"]
            if failure == "parity": evidence["passed"] = False
            if failure == "zero_branch": evidence["nonzero_branch"] = False
            if failure == "gradients": evidence["gradients"]["passed"] = False
            if failure == "backward": evidence["gradients"]["checkpoint_backward"] = False
            if failure == "trained_hash": evidence["trained_checkpoint"]["checkpoint_sha256"] = "different"
            if failure == "events": evidence["trained_checkpoint"]["events"] = 1
        hashed(row)
        if failure == "benchmark_hash": row["projected_training_hours"] = 5
        P.write_json(path,row)
    with pytest.raises((ValueError, FileNotFoundError)):
        P.benchmark_summary(tmp_path,spec)


def test_approval_includes_speed_evidence_and_refuses_silent_change(tmp_path):
    spec = fixture(tmp_path)
    with pytest.raises(ValueError, match="reviewed"):
        P.approve(tmp_path,spec,reviewed=False)
    P.approve(tmp_path,spec,reviewed=True)
    P.require_approval(tmp_path,spec)
    approval = P.read_json(tmp_path/"production_approval.json")
    assert len(approval["benchmark_sha256"]) == 6
    path = tmp_path/"benchmarks"/(P.TASKS[0]["run_id"]+"_speed.json")
    row = P.read_json(path)
    row["models"]["EV"]["512"].update(summarize_samples([.2,.3,.4],512))
    P.write_json(path,hashed(row))
    with pytest.raises(ValueError, match="admission"):
        P.require_approval(tmp_path,spec)


def test_frozen_source_state_reference_and_launch(tmp_path,monkeypatch):
    donor = tmp_path/"donor"
    d = hashed(dict(content="fixture"))
    P.write_json(donor/"campaign.json",d)
    inventory = dict(splits={})
    inventory["inventory_sha256"] = P.digest(inventory)
    contract = dict(path=str(donor),campaign_sha256=d["content_hash"])
    state = tmp_path/"state"
    state.mkdir()
    cp = state/"best.pt"
    cp.write_bytes(b"fixture")
    reference = dict(path=str(state),checkpoint=str(cp), checkpoint_sha256=P.file_sha256(cp),
                     artifact_sha256={"best.pt":P.file_sha256(cp)})
    monkeypatch.setattr(P,"state_reference",lambda *a: (dict(inventory=inventory),{},reference))
    contents, launch = dict(files="fixture"),dict(partition="tier3",state_donor=str(state))
    spec = P.create_spec(inventory,launch,contract,contents)
    for name,row in (("campaign.json",spec),("launch.json",launch),("data_contents.json",contents)):
        P.write_json(tmp_path/name,row)
    assert P.load_campaign(tmp_path)["tasks"] == P.TASKS
    assert "part_ev_state_speed/model.py" in spec["source"]
    assert "part_ev_message_study/model.py" in spec["source"]
    cp.write_bytes(b"changed")
    with pytest.raises(ValueError,match="reference artifact"):
        P.load_campaign(tmp_path)
