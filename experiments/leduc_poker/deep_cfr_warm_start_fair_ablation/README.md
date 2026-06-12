# Leduc Poker Deep CFR Fair Warm-Start Ablation

This experiment tests whether interrupting Deep CFR at an intermediate
checkpoint, saving the full solver state, and resuming from that checkpoint
changes learning behaviour relative to a continuous run.

The experiment is deliberately paired and controlled. For each seed, the
continuous baseline and warm-start arm use the same Leduc poker environment,
Deep CFR implementation, architecture, optimiser settings, replay-buffer
settings, traversal count, policy-network training frequency, evaluation
interval, and total training budget. The intended treatment is only the
checkpoint/resume mechanism.

## Research question

Under matched compute and matched random seeds, does saving and reloading the
full Deep CFR solver state at an intermediate boundary preserve the learning
trajectory of the continuous baseline?

## Protocol

Default arms:

| Arm | Description |
| --- | --- |
| `baseline_continuous` | Train continuously from random initialisation for 1,500 iterations. |
| `warm_start` | Train for 100 iterations, save a full checkpoint, reload into a fresh solver, and train for the remaining 1,400 iterations. |

The main paired quantity is:

```text
warm-start metric - baseline metric
```

Positive values mean the warm-start arm is worse for error-like metrics such
as exploitability; negative values mean it is better.

Run from the repository root:

```bash
python -m experiments.leduc_poker.deep_cfr_warm_start_fair_ablation.run
```

Quick smoke run:

```bash
python -m experiments.leduc_poker.deep_cfr_warm_start_fair_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --warm-start-boundary 1 \
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

The default uses five paired seeds. Use
`--seeds 1234,2025,31415,27182,16180,4242,8675309,7,99,1001` for the full
ten-seed baseline list if runtime permits.

## Outputs

Outputs are written to:

```text
outputs/leduc_poker_deep_cfr_fair_warm_start_ablation_<YYYYMMDD_HHMMSS>/
```

Key artefacts:

| File | Contents |
| --- | --- |
| `experiment_metadata.json` | Full config, seed lists, software versions, and output references. |
| `experiment.log` | Full log mirroring stdout. |
| `seed_summary.csv` | One row per `(seed, arm)` run. |
| `checkpoint_curves.csv` | Per-checkpoint curves for both arms. |
| `aggregate_summary.json` | Mean, standard deviation, standard error, and count by arm. |
| `paired_summary.csv` | Final per-seed warm-start minus baseline differences. |
| `paired_aggregate_summary.json` | Aggregate paired-difference statistics. |
| `paired_checkpoint_differences.csv` | Per-checkpoint paired differences by seed. |
| `ablation_curves.npz` | Compact matrices for downstream thesis plotting. |
| `checkpoints/seed_<seed>_iter_<boundary>_full.pt` | Full checkpoint saved by the warm-start arm. |
| `failed_seeds.json` | Present only if one or more arm runs fail. |

Generated PNGs:

| File | Figure |
| --- | --- |
| `exploitability_by_iteration.png` | Baseline and warm-start exploitability curves. |
| `exploitability_by_nodes.png` | Exploitability by nodes touched. |
| `average_policy_value_by_iteration.png` | Baseline and warm-start average-policy value curves. |
| `average_policy_value_by_nodes.png` | Average-policy value by nodes touched. |
| `policy_value_error_by_iteration.png` | Policy-value error from the known Leduc value. |
| `paired_delta_exploitability_warm_minus_baseline.png` | Paired exploitability difference over training. |
| `paired_difference_summary_bar_chart.png` | Mean paired differences for final, best, and AUC metrics. |

Exploitability is the primary strategic metric. A large paired difference is
evidence that the checkpoint/resume mechanism changes the learning process,
not evidence that the warm-start arm received extra training.
