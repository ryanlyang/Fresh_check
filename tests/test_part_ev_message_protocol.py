import copy
import pytest
from part_ev_message_study import protocol as P, controls as C
from part_ev_screen import protocol as S
from tests.test_part_ev_screen_protocol import precision_record


def test_five_fresh_tasks_exact_donor_recipe_and_no_original_mutation():
    before = copy.deepcopy(S.TASKS)
    assert len(P.TASKS) == len({t["run_id"] for t in P.TASKS}) == 5
    assert not set(C.CONTROL_IDS).intersection(t["run_id"] for t in P.TASKS)
    assert {t["seed"] for t in P.TASKS} == {101}
    assert P.recipes() == {"ranger_short_v1":S.recipes()["ranger_short_v1"]}
    for step in (0,139999,140000,195999,196000,199999):
        assert P.learning_rate(step,P.recipes()["ranger_short_v1"]) == S.learning_rate(step,S.recipes()["ranger_short_v1"])
    assert S.TASKS == before
    for bad in (-1,5,None,True):
        with pytest.raises(ValueError):
            P.task_directory("/campaign",bad)


def benchmark_fixture(root):
    donor = root/"donor"
    spec = dict(content_hash="campaign", donor=dict(path=str(donor),
                runtime=dict(gpu="A100",gpu_memory_bytes=1000)))
    runtime = dict(spec["donor"]["runtime"],campaign_sha256="campaign")
    P.write_json(donor/"benchmarks/EV_RANGER.json",dict(input_fingerprints={"0":"same","20":"same20"}))
    for task in P.TASKS:
        row = dict(task=task,campaign_sha256="campaign",runtime=runtime,passed=True,
            measured_steps=4200,amp_skipped_steps=2, initial_parity=dict(passed=True),
            training_precision=precision_record("torch.float16"),
            validation=dict(precision=precision_record("torch.float32"),events=128000,seconds=100),
            input_fingerprints={"0":"same","20":"same20"},
            native_peak_bytes=100,full_length_peak_bytes=200,validation_peak_bytes=300,
            projected_training_hours=10,
            full_length_attempts=[dict(attempt=i+1,loss=1.,state_finite=True,
                                      amp_skipped=i==0,gradient_norm=None if i==0 else 2.) for i in range(18)])
        row["content_hash"] = P.digest(row)
        P.write_json(root/"benchmarks"/(task["run_id"]+".json"),row)
    return spec,runtime


@pytest.mark.parametrize("failure",["missing","hash","task","runtime","donor_runtime","input","donor_input",
    "precision","val_precision","parameters","memory","confirm_skip","cal_last_skip","state_nan",
    "incomplete","skip_rate","time","val_events"])
def test_benchmark_evidence_fail_closed(tmp_path,failure):
    spec,runtime = benchmark_fixture(tmp_path)
    assert P.benchmark_summary(tmp_path,spec,runtime)["recommended_train_hours"] == 13
    path = tmp_path/"benchmarks/EV_STATE_RANGER.json"
    row = P.read_json(path)
    if failure == "missing":
        path.unlink()
    else:
        if failure == "task": row["task"]["seed"] = 202
        if failure == "runtime": row["runtime"]["gpu"] = "H100"
        if failure == "donor_runtime": spec["donor"]["runtime"]["gpu"] = "H100"
        if failure == "input": row["input_fingerprints"]["0"] = "changed"
        if failure == "donor_input":
            P.write_json(tmp_path/"donor/benchmarks/EV_RANGER.json",dict(input_fingerprints={"0":"changed"}))
        if failure == "precision": row["training_precision"]["passed"] = False
        if failure == "val_precision": row["validation"]["precision"]["observed_activations"] = {}
        if failure == "parameters": row["training_precision"]["parameter_dtypes"] = ["torch.float16"]
        if failure == "memory": row["validation_peak_bytes"] = 901
        if failure == "confirm_skip": row["full_length_attempts"][-1]["amp_skipped"] = True
        if failure == "cal_last_skip": row["full_length_attempts"][7]["amp_skipped"] = True
        if failure == "state_nan": row["full_length_attempts"][0]["state_finite"] = False
        if failure == "incomplete": row["full_length_attempts"].pop()
        if failure == "skip_rate": row["amp_skipped_steps"] = 210
        if failure == "time": row["projected_training_hours"] = -1
        if failure == "val_events": row["validation"]["events"] = 100
        row["content_hash"] = P.digest({k:v for k,v in row.items() if k != "content_hash"})
        if failure == "hash": row["passed"] = False
        P.write_json(path,row)
    with pytest.raises((ValueError,FileNotFoundError)):
        P.benchmark_summary(tmp_path,spec,runtime)


def test_explicit_approval_freezes_benchmark_evidence(tmp_path):
    spec,_ = benchmark_fixture(tmp_path)
    with pytest.raises(ValueError,match="reviewed"):
        P.approve(tmp_path,spec,reviewed=False)
    assert not (tmp_path/"production_approval.json").exists()
    P.approve(tmp_path,spec,reviewed=True)
    P.require_approval(tmp_path,spec)
    path = tmp_path/"benchmarks/EV_STATE_RANGER.json"
    row = P.read_json(path)
    row["projected_training_hours"] = 12
    row["content_hash"] = P.digest({k:v for k,v in row.items() if k != "content_hash"})
    P.write_json(path,row)
    with pytest.raises(ValueError,match="admission"):
        P.require_approval(tmp_path,spec)


def test_campaign_source_recipe_data_environment_and_launch_are_frozen(tmp_path):
    donor = tmp_path/"donor"
    d = dict(content="fixture")
    d["content_hash"] = P.digest(d)
    P.write_json(donor/"campaign.json",d)
    inventory = dict(splits={})
    inventory["inventory_sha256"] = P.digest(inventory)
    contract = dict(path=str(donor),campaign_sha256=d["content_hash"])
    contents,launch = dict(files="fixture"),dict(partition="tier3")
    spec = P.create_spec(inventory,launch,contract,contents)
    P.write_json(tmp_path/"campaign.json",spec)
    P.write_json(tmp_path/"launch.json",launch)
    P.write_json(tmp_path/"data_contents.json",contents)
    assert P.load_campaign(tmp_path)["tasks"] == P.TASKS
    spec["tasks"][0]["seed"] = 202
    spec["content_hash"] = P.digest({k:v for k,v in spec.items() if k != "content_hash"})
    P.write_json(tmp_path/"campaign.json",spec)
    with pytest.raises(ValueError,match="tasks"):
        P.load_campaign(tmp_path)


def test_metadata_guard_detects_changed_inputs(tmp_path):
    path = tmp_path/"data.root"
    path.write_bytes(b"original")
    stat = path.stat()
    spec = dict(inventory=dict(splits=dict(train=dict(files=[dict(path=str(path),
        size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)]),val=dict(files=[]))))
    P.verify_inventory_metadata(spec)
    path.write_bytes(b"modified data")
    with pytest.raises(ValueError,match="changed"):
        P.verify_inventory_metadata(spec)
