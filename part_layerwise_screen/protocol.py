"""Direct-submit protocol; no invented benchmark or reviewed admission evidence."""
import copy
import json
from pathlib import Path
from part_ev_fullscale.protocol import read_json, write_json, file_sha256, file_lock
from part_ev_screen.protocol import recipes, runtime_environment
from part_ev_screen.precision import POLICY
from part_screen_common.contracts import checked_json, hashed, donor_contract, verify_dataset_metadata, runtime_identity
from .model import COUNTS

SCHEMA = "part_layerwise_screen_v1"
CONTROLS = ["PART_RANGER", "EV_RANGER"]
TASKS = [dict(run_id=name, architecture=arch, recipe="ranger_short_v1", seed=101)
    for name, arch in (("PART_LAYERWISE_RANGER", "part_layerwise"), ("EV_LAYERWISE_RANGER", "part_ev_layerwise"))]
CONFIG = dict(pair_stem="native_shared_64", bias_heads="eight_independent_native_conv_bn_gelu_heads",
    initialization="identical_copies_of_shared_bias_head", edgevalue="unchanged_per_layer_projections",
    selected_relations=False)
ADMISSION = dict(mode="direct_training_without_separate_preflight", benchmark_performed=False,
    startup_checks="fresh_discarded_probe_input_identity_initial_parity_fp16_step_fp32_forward",
    production_initialization="fresh_seeded_model_optimizer_scaler")


def environment_differences(expected, actual, prefix=""):
    """Describe mismatches without dropping any part of the donor contract."""
    differences = []
    for key in sorted(set(expected) | set(actual)):
        field = prefix + key
        if key not in expected or key not in actual:
            differences.append(dict(field=field, expected=expected.get(key, "<missing>"),
                                    actual=actual.get(key, "<missing>")))
        elif isinstance(expected[key], dict) and isinstance(actual[key], dict):
            differences.extend(environment_differences(expected[key], actual[key], field + "."))
        elif expected[key] != actual[key]:
            differences.append(dict(field=field, expected=expected[key], actual=actual[key]))
    return differences


def require_environment(expected):
    differences = environment_differences(expected, runtime_environment())
    if differences:
        details = json.dumps(differences[:20], sort_keys=True)
        raise ValueError(f"Environment differs from the donor ({len(differences)} fields): {details}. "
                         "Use the unchanged donor-matched SPORC Python environment.")


def task_directory(output, index):
    if type(index) is not int or not 0 <= index < len(TASKS):
        raise ValueError("Only new-model task indices 0 and 1 are valid")
    task = TASKS[index]
    return Path(output) / "runs" / task["run_id"] / f"seed_{task['seed']}"


def source_manifest(root):
    root = Path(root)
    paths = []
    for package in ("part_layerwise_screen", "part_screen_common", "part_ev_screen", "part_ev_fullscale"):
        paths.extend((root / package).glob("*.py"))
    paths.extend(root / name for name in ("scripts/run_part_layerwise_screen.py", "scripts/run_part_ev_screen.py",
        "sbatch/run_part_layerwise_screen_sporc.sh", "sbatch/submit_part_layerwise_screen_sporc.sh",
        "particle_transformer/data/JetClass/JetClass_full.yaml"))
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(paths)}


def create_spec(source, launch):
    contract = donor_contract(launch["donor"], CONTROLS, local_source=source)
    require_environment(contract["environment"])
    verify_dataset_metadata(contract["inventory"])
    return hashed(dict(schema=SCHEMA, tasks=copy.deepcopy(TASKS), model=CONFIG, parameters=COUNTS,
        recipes={"ranger_short_v1": contract["recipes"]["ranger_short_v1"]}, donor=contract,
        environment=contract["environment"], inventory=contract["inventory"], precision=POLICY,
        admission=ADMISSION, launch=launch, source=source_manifest(source)))


def load_campaign(output, *, check_environment=True):
    spec = checked_json(Path(output) / "experiment_spec.json", SCHEMA)
    if (spec["tasks"] != TASKS or spec["model"] != CONFIG or spec["parameters"] != COUNTS
            or spec["admission"] != ADMISSION or spec["precision"] != POLICY
            or spec["donor"]["controls"] != CONTROLS
            or spec["recipes"] != {"ranger_short_v1": recipes()["ranger_short_v1"]}):
        raise ValueError("Frozen layerwise protocol differs")
    if spec["source"] != source_manifest(Path(__file__).resolve().parents[1]):
        raise ValueError("Use the unchanged frozen source snapshot")
    if check_environment:
        require_environment(spec["environment"])
    return spec


def require_approval(output, spec, runtime=None):
    # ScreenEngine's admission interface. This deliberately is NOT a benchmark
    # approval: the user's direct-submit choice is explicit in the campaign.
    if spec.get("admission") != ADMISSION:
        raise ValueError("Missing direct-submit admission policy")
    if runtime is not None and runtime_identity(runtime) != spec["donor"]["runtime"]:
        raise ValueError("Runtime differs from donor controls")
    return ADMISSION
