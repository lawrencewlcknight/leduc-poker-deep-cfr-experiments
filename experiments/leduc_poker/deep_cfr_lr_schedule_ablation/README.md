# Leduc Poker Deep CFR Learning-Rate Schedule Ablation

This experiment tests whether a learning-rate schedule improves the stability
or final performance of the shared Deep CFR implementation in Leduc poker. It is
aligned with the Experiment 2 baseline configuration: game, CFR budget,
traversals, network architecture, replay memory, batch sizes, policy-network
training cadence, and seeds are held fixed. The intended treatment variable is
only the learning-rate schedule.

The default protocol compares:

| Schedule id | Description |
| --- | --- |
| `constant_baseline_exp2` | Constant `0.003` learning rate, matching the Experiment 2 baseline. |
| `cosine_decay_to_10pct` | Cosine decay from `0.003` to `0.0003`. |

Optional exploratory schedules can be enabled with
`--include-optional-schedules`: `linear_decay_to_10pct` and
`step_decay_to_10pct`.

## Run

From the repository root:

```bash
python -m experiments.leduc_poker.deep_cfr_lr_schedule_ablation.run
```

Quick smoke test:

```bash
python -m experiments.leduc_poker.deep_cfr_lr_schedule_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
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

- `--schedule-ids constant_baseline_exp2,cosine_decay_to_10pct` selects a subset.
- `--include-optional-schedules` adds the linear and step-decay arms.
- `--learning-rate`, `--learning-rate-end`, `--learning-rate-decay-rate`,
  `--learning-rate-decay-steps`, and `--learning-rate-warmup-iterations`
  override schedule parameters.
- `--save-final-checkpoints` writes final full-model checkpoints for every
  seed and schedule.

## Outputs

The run directory follows `docs/OUTPUT_CONVENTIONS.md`:

- `seed_summary.csv` — one row per `(schedule, seed)`, including final
  exploitability, policy-value error, AUC metrics, final learning rate, and
  policy-training update counts.
- `checkpoint_curves.csv` — one row per checkpoint with `schedule`,
  `learning_rate_schedule`, actual `learning_rate`, exploitability,
  policy-value error, replay-buffer sizes, losses, gradient norms, and policy
  diagnostics.
- `aggregate_summary.json` — across-seed mean / standard deviation / standard
  error grouped by schedule.
- `paired_differences_vs_baseline.csv` — per-seed deltas for each non-baseline
  schedule minus `constant_baseline_exp2`.
- `paired_difference_summary.json` — aggregate paired differences and the
  fraction of paired seeds for which the schedule improved each metric.
- `ablation_curves.npz` — per-schedule arrays for exploitability,
  policy-value error, nodes touched, wall-clock, and learning rate.
- `experiment_metadata.json` and `experiment.log` — full reproducibility
  metadata and run log.
- PNG plots:
  - `learning_rates.png`
  - `exploitability_by_iteration.png`
  - `exploitability_by_nodes.png`
  - `average_policy_value_by_iteration.png`
  - `average_policy_value_by_nodes.png`
  - `policy_value_error_by_iteration.png`
  - `final_exploitability_by_schedule.png`
  - `final_average_policy_value_by_schedule.png`
  - `delta_final_exploitability_vs_baseline.png`
  - `policy_loss_diagnostic.png`
  - `advantage_target_variance_diagnostic.png`
  - `policy_entropy_diagnostic.png`

Paired differences are reported as `schedule - baseline`; negative
exploitability, AUC, or policy-value-error deltas are improvements.
