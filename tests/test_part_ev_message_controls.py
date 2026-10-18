import copy
import shutil
from pathlib import Path
import numpy as np
import pytest
import torch
from part_ev_message_study import controls as C, protocol as P
from part_ev_screen import protocol as S


def training_fixture(root):
    task = dict(S.TASKS[0])
    runtime = dict(gpu="A100",campaign_sha256="fixture")
    spec = dict(content_hash="fixture",recipes=P.recipes(),parameters=S.COUNTS,tasks=[task])
    run = root/"runs/PART_RANGER/seed_101"
    checks = []
    for i in range(1,11):
        check = dict(check=i,step=20000*i,events=1280000,accuracy=.8+min(i,9)/1000,
                     macro_auc_ovo=.98,macro_auc_ovr=.99,loss=.4)
        P.write_json(run/"validation"/f"check_{i:02d}.json",check)
        checks.append(check)
    checkpoint = run/"best_model_val.pt"
    torch.save(dict(task=task,campaign_sha256="fixture",runtime=runtime,
                    validation=checks[8],model={"fixture":torch.ones(1)}),checkpoint)
    complete = dict(task=task,step=200000,validated_checks=10,best_check=9,
                    best_accuracy=checks[8]["accuracy"],campaign_sha256="fixture",
                    runtime=runtime,best_checkpoint_sha256=P.file_sha256(checkpoint))
    P.write_json(run/"training_complete.json",complete)
    P.write_json(run/"optimizer.json",dict(optimizer="Ranger",parameters=S.COUNTS["part"],
        groups=[dict(weight_decay=0,betas=[.95,.999],eps=1e-5)]))
    P.write_json(root/"campaign.json",spec)
    return spec,task,runtime,run


@pytest.mark.parametrize("failure",["budget","checks","tie","hash","runtime","checkpoint_metadata","optimizer","missing"])
def test_completed_control_validated_and_corruption_rejected(tmp_path,failure):
    spec,task,runtime,run = training_fixture(tmp_path)
    identity = C.runtime_identity(runtime)
    assert C.validate_training(tmp_path,spec,task,identity)["best_check"] == 9
    complete = P.read_json(run/"training_complete.json")
    if failure == "budget": complete["step"] = 180000
    if failure == "checks": complete["validated_checks"] = 9
    if failure == "tie": complete["best_check"] = 10
    if failure == "hash": complete["best_checkpoint_sha256"] = "wrong"
    if failure == "runtime": complete["runtime"]["gpu"] = "GH200"
    if failure == "checkpoint_metadata":
        checkpoint = C.trusted_load(run/"best_model_val.pt")
        checkpoint["task"]["seed"] = 202
        torch.save(checkpoint,run/"best_model_val.pt")
        complete["best_checkpoint_sha256"] = P.file_sha256(run/"best_model_val.pt")
    if failure == "optimizer":
        optimizer = P.read_json(run/"optimizer.json")
        optimizer["groups"][0]["weight_decay"] = .01
        P.write_json(run/"optimizer.json",optimizer)
    if failure == "missing": (run/"validation/check_03.json").unlink()
    P.write_json(run/"training_complete.json",complete)
    with pytest.raises((ValueError,FileNotFoundError)):
        C.validate_training(tmp_path,spec,task,identity)


def test_pending_controls_allowed_for_training_but_never_for_final_report(tmp_path):
    spec = dict(tasks=S.TASKS)
    contract = dict(runtime={},path=str(tmp_path))
    before = list(tmp_path.iterdir())
    registry,labels = C.registry(tmp_path,spec,contract)
    assert all(r["status"] == "pending_training" for r in registry["controls"].values())
    assert not labels and list(tmp_path.iterdir()) == before
    with pytest.raises(ValueError,match="pending"):
        C.registry(tmp_path,spec,contract,require_complete=True)


def test_missing_evaluation_is_pending_not_fabricated(tmp_path,monkeypatch):
    spec,task,runtime,_ = training_fixture(tmp_path)
    monkeypatch.setattr(C,"CONTROL_IDS",("PART_RANGER",))
    registry,_ = C.registry(tmp_path,spec,dict(runtime=C.runtime_identity(runtime)))
    row = registry["controls"]["PART_RANGER"]
    assert row["status"] == "pending_evaluation" and "metrics" not in row


def receipt_fixture(root):
    inventory = dict(data_root=str(root),splits=dict(train=dict(files=[]),val=dict(files=[])))
    for i in range(11):
        split, folder = ("train","train_100M") if i < 10 else ("val","val_5M")
        file = root/"Pythia"/folder/f"fixture_{i}.root"
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_bytes(f"ROOT fixture {i}".encode())
        stat = file.stat()
        inventory["splits"][split]["files"].append(dict(path=str(file),size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
        archive = f"JetClass_Pythia_train_100M_part{i}.tar" if i < 10 else "JetClass_Pythia_val_5M.tar"
        P.write_json(root/"receipts"/(archive+".json"),dict(contract="jetclass_archive_stage_v1",archive=archive,
            members=[dict(path=file.relative_to(root).as_posix(),size=stat.st_size,sha256=P.file_sha256(file))]))
    return inventory


def test_staging_receipts_verify_bytes_not_just_file_size(tmp_path):
    inventory = receipt_fixture(tmp_path)
    assert len(C.verify_data_contents(inventory)["file_sha256"]) == 11
    path = Path(inventory["splits"]["train"]["files"][0]["path"])
    original = path.read_bytes()
    path.write_bytes(b"X"+original[1:])
    with pytest.raises(ValueError,match="content"):
        C.verify_data_contents(inventory)


def test_donor_source_contract_is_read_only_and_fails_on_mismatch(tmp_path,monkeypatch):
    inventory = dict(splits={})
    inventory["inventory_sha256"] = P.digest(inventory)
    spec = S.create_spec(inventory,C.ROOT,{"fixture":True})
    P.write_json(tmp_path/"campaign.json",spec)
    P.write_json(tmp_path/"launch.json",spec["launch"])
    for name in spec["source"]:
        path = tmp_path/"source"/name
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/name,path)
    monkeypatch.setattr(S,"benchmark_summary",lambda *a:dict(passed=True,runtime=dict(gpu="A100",
        campaign_sha256=spec["content_hash"])))
    for task in S.TASKS:
        P.write_json(tmp_path/"benchmarks"/(task["run_id"]+".json"),dict(fixture=True))
    before = {str(p.relative_to(tmp_path)):P.file_sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    _,contract = C.donor_contract(tmp_path)
    assert contract["runtime"] == {"gpu":"A100"}
    after = {str(p.relative_to(tmp_path)):P.file_sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    (tmp_path/"source/part_ev_screen/model.py").write_text("# changed")
    with pytest.raises(ValueError,match="source"):
        C.donor_contract(tmp_path)


def evaluation_fixture(root):
    """Real compact prediction files with the full required label population."""
    spec,task,runtime,run = training_fixture(root)
    trained = C.validate_training(root,spec,task,C.runtime_identity(runtime))
    selected = dict(task=task,checkpoint=str(run/"best_model_val.pt"),sha256=trained["checkpoint_sha256"])
    P.write_json(root/"selection/locked_checkpoints.json",dict(campaign_sha256="fixture",checkpoints=[selected]))
    files,records = [],[]
    directory = run/"full_validation"
    directory.mkdir()
    for i in range(10):
        file = dict(name=f"fixture_{i}.root",num_entries=500000,path=f"/data/fixture_{i}.root")
        files.append(file)
        path = directory/f"file_{i:03d}.npz"
        np.savez_compressed(path,labels=np.full(500000,i,dtype=np.int8),
                            logits=np.zeros((500000,10),dtype=np.float32))
        record = dict(sha256=P.file_sha256(path),contract=dict(campaign_sha256="fixture",
            checkpoint_sha256=trained["checkpoint_sha256"],file=file,entry_start=0,entry_stop=500000))
        records.append(record)
        P.write_json(path.with_suffix(".json"),record)
    spec["inventory"] = dict(splits=dict(val=dict(files=files)))
    row = dict(task=task,campaign_sha256="fixture",checkpoint_sha256=trained["checkpoint_sha256"],
               identity_manifest=records,metrics=dict(events=5000000,accuracy=.1))
    P.write_json(directory/"metrics.json",row)
    return spec,task,trained,directory


def test_complete_prediction_population_and_sidecars(tmp_path):
    spec,task,trained,directory = evaluation_fixture(tmp_path)
    row,truth = C.validate_evaluation(tmp_path,spec,task,trained)
    assert row["status"] == "complete" and len(truth) == 5000000
    sidecar = directory/"file_000.json"
    changed = P.read_json(sidecar)
    changed["contract"]["entry_start"] = 1
    P.write_json(sidecar,changed)
    with pytest.raises(ValueError,match="sidecar"):
        C.validate_evaluation(tmp_path,spec,task,trained)


def test_metrics_must_match_recomputation(tmp_path):
    spec,task,trained,_ = evaluation_fixture(tmp_path)
    with pytest.raises(ValueError,match="recomputed"):
        C.validate_evaluation(tmp_path,spec,task,trained,recompute=lambda *a,**kw:dict(events=5000000,accuracy=.9))
