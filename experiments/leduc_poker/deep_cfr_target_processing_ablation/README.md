# Leduc Poker Deep CFR Target-Processing Ablation

This experiment tests whether simple preprocessing of sampled advantage
targets improves Deep CFR performance in Leduc poker. The replay buffers store
raw sampled instantaneous regret targets for every variant; processing is
applied only to the supervised advantage-network fitting loss.

The default variants are:

| Variant id | Processing |
| --- | --- |
| `raw_targets_exp2_baseline` | Raw Experiment 2-aligned targets. |
| `standardized_targets` | Batch-standardized advantage targets. |
| `clipped_targets` | Targets clipped to `[-1.0, 1.0]`. |
| `standardized_clipped_targets` | Batch-standardized, then clipped to `[-1.0, 1.0]`. |

All other core training settings are aligned with the Experiment 2 Deep CFR
baseline: OpenSpiel Leduc poker, 1,500 CFR iterations, 320 traversals per
iteration, 32x32 networks, replay-memory capacity, batch sizes, policy-training
cadence, and evaluation interval.

## Run

From the repository root:

```bash
python -m experiments.leduc_poker.deep_cfr_target_processing_ablation.run
```

Quick smoke test:

```bash
python -m experiments.leduc_poker.deep_cfr_target_processing_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids raw_targets_exp2_baseline,standardized_clipped_targets \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 0 \
  --batch-size-strategy 0 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests
```

Useful CLI options:

- `--variant-ids` selects a comma-separated subset of target-processing arms.
- `--target-clip-value` changes the clipping threshold for clipping variants.
- `--target-standardize-epsilon` changes the minimum scale used for
  standardisation.
- `--save-final-checkpoints` writes final full-model checkpoints for every
  seed and variant.

## Outputs

The run directory follows `docs/OUTPUT_CONVENTIONS.md`:

- `seed_summary.csv` — one row per `(variant_id, seed)`, including
  exploitability, policy-value error, raw target variance, processed target
  variance, clipping fraction, and policy update counts.
- `checkpoint_curves.csv` — one row per checkpoint with `variant_id`,
  `target_processing`, exploitability, value diagnostics, raw target variance,
  processed target variance by player, standardisation scale by player, clip
  fraction by player, losses, replay sizes, and gradient norms.
- `aggregate_summary.json` — across-seed mean / standard deviation / standard
  error grouped by target-processing variant.
- `paired_differences_vs_baseline.csv` — per-seed target-processing variant
  minus raw-target baseline deltas.
- `paired_difference_summary.json` — aggregate paired differences and the
  fraction of paired seeds where each variant improved each metric.
- `ablation_curves.npz` — per-variant arrays for exploitability,
  policy-value error, nodes touched, wall-clock, processed target variance,
  and clip fraction.
- PNG plots:
  - `exploitability_by_iteration.png`
  - `exploitability_by_nodes.png`
  - `average_policy_value_by_iteration.png`
  - `average_policy_value_by_nodes.png`
  - `policy_value_error_by_iteration.png`
  - `final_exploitability_by_variant.png`
  - `final_average_policy_value_by_variant.png`
  - `paired_deltas_vs_baseline.png`
  - `processed_target_variance.png`
  - `policy_loss_diagnostic.png`

Paired differences are reported as `variant - raw baseline`; negative
exploitability, AUC, or policy-value-error deltas are improvements.
