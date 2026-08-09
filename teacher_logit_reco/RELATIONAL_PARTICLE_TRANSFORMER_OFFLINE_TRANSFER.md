# Relational Particle Transformer Offline Transfer

This campaign tests whether the HLT-selected edge-conditioned-value result transfers to the offline JetClass view. It is a separate campaign: it does not alter the completed HLT campaign or reuse its run IDs.

## Frozen matrix

The campaign trains each row from scratch at seeds 101, 202, and 303:

| Offline run ID | Relations | Attention path |
|---|---|---|
| `OFF_RPT_BASE` | standard four ParT pair features | ordinary shared pair bias |
| `OFF_RPT_BASE_EDGEVALUE` | standard four ParT pair features | edge-conditioned values |
| `OFF_RPT_SELECTED_LAYERWISE` | standard four + PT + TRACK + REGION | layerwise pair bias |
| `OFF_RPT_SELECTED_EDGEVALUE` | standard four + PT + TRACK + REGION | edge-conditioned values |

The primary comparison is `OFF_RPT_SELECTED_EDGEVALUE - OFF_RPT_BASE`. The base-edgevalue row separates the attention-path effect from the selected-relation effect; the selected-layerwise row separates edge-conditioned values from layerwise relation injection.

## Data and selection discipline

- Offline events, labels, identity order, and four split memberships must exactly match the completed HLT parent campaign.
- Non-tree and REGION normalizers are refit using offline `model_train` only.
- Checkpoints use the original RPT production optimizer, schedule, early-stopping, and `model_val` selection protocol.
- `stack_val` reports validation results but cannot prune any of the twelve predeclared final-test tasks.
- An immutable lock containing all twelve checkpoints is written before `final_test` is opened.
- There is no performance threshold. Poor accuracy or rejection never cancels or removes a run. Runtime and integrity errors still fail the affected job rather than publishing unauthenticated science.

The final result is an offline-domain replication. It is not described as a globally untouched offline test, because these offline split identities may have been used by older experiments in this repository.

## Submission

On Tigris, after committing and pulling this implementation:

```bash
cd /home/ryreu/atlas/Fresh_check
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export RPT_OFFLINE_PARENT_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/relational_particle_transformer/rpt_attention_bias_20260729T155648Z_bfcda5bd6f_049b2555e6"
bash sbatch/submit_relational_part_offline_transfer.sh --dry-run
bash sbatch/submit_relational_part_offline_transfer.sh
```

Concurrency can be changed before submission with `RPT_OFFLINE_TREE_CONCURRENCY`, `RPT_OFFLINE_TRAIN_CONCURRENCY`, and `RPT_OFFLINE_FINAL_CONCURRENCY`. This affects scheduling only, not scientific meaning.

The final artifacts are `reports/offline_transfer_report.json` and `reports/offline_transfer_report.md` under the printed campaign root.
