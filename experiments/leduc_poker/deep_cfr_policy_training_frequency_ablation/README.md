# Leduc Poker Deep CFR Policy-Training-Frequency Ablation

This experiment tests whether the cadence of average-policy-network training
affects Deep CFR performance in Leduc poker.

Deep CFR stores sampled average strategies in policy memory and fits a separate
average-policy network from that memory. In the baseline experiments this
network is trained periodically during CFR so exploitability and policy-value
diagnostics can be computed throughout the run. This ablation treats the
training frequency as the experimental variable.

## Research question

Holding CFR iterations, traversals, seeds, architecture, optimiser settings,
memory capacity, advantage-network training, and evaluation checkpoints fixed,
does training the average-policy network more frequently improve the measured
quality or stability of the evaluated average policy?

## Protocol

The default ablation arms are:

```text
policy_network_train_every = 10, 25, 50, 100
```

The reference arm for paired comparisons is `10`, the most frequent
policy-network training regime. Evaluation remains fixed at every 25 CFR
iterations for all arms, so curves are directly comparable across frequencies.

Run from the repository root:

```bash
python -m experiments.leduc_poker.deep_cfr_policy_training_frequency_ablation.run
```

Quick smoke run:

```bash
python -m experiments.leduc_poker.deep_cfr_policy_training_frequency_ablation.run \
  --seeds 1234 \
  --iterations 25 \
  --traversals 8 \
  --evaluation-interval 5 \
  --policy-train-every-variants 1,5 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --output-root outputs/smoke_tests
```

Use `--seeds 1234,2025,31415,27182,16180,4242,8675309,7,99,1001` for the
extended ten-seed thesis run if compute budget permits.

## Outputs

Outputs are written to:

```text
outputs/leduc_poker_deep_cfr_policy_training_frequency_ablation_<YYYYMMDD_HHMMSS>/
```

Key artefacts:

| File | Contents |
| --- | --- |
| `experiment_metadata.json` | Full config, seed lists, software versions, and output references. |
| `experiment.log` | Full log mirroring stdout. |
| `seed_summary.csv` | One row per `(policy_network_train_every, seed)` run. |
| `checkpoint_curves.csv` | Per-checkpoint curves with policy-training frequency, exploitability, value error, losses, replay sizes, and diagnostics. |
| `aggregate_summary.json` | Mean, standard deviation, standard error, and finite count by policy-training frequency. |
| `paired_differences_vs_reference.csv` | Per-seed deltas versus the reference frequency. |
| `ablation_curves.npz` | Compact matrices for downstream thesis plotting. |
| `failed_seeds.json` | Present only if one or more runs fail. |

Generated PNGs:

| File | Figure |
| --- | --- |
| `exploitability_by_iteration.png` | Mean exploitability curves by policy-training frequency. |
| `exploitability_by_nodes.png` | Mean exploitability by nodes touched. |
| `average_policy_value_by_iteration.png` | Mean average-policy value curves by policy-training frequency. |
| `average_policy_value_by_nodes.png` | Mean average-policy value by nodes touched. |
| `policy_value_error_by_iteration.png` | Mean policy-value error from the known Leduc value. |
| `final_exploitability_by_frequency.png` | Final exploitability bar chart. |
| `final_average_policy_value_by_frequency.png` | Final average-policy value bar chart. |
| `stability_summaries_by_frequency.png` | Final-window exploitability and normalised AUC summaries. |
| `policy_loss_by_frequency.png` | Average-policy supervised loss diagnostic. |
| `cumulative_policy_training_events.png` | Cumulative policy-training events by arm. |
| `policy_entropy_by_frequency.png` | Average-policy entropy diagnostic. |
| `paired_final_exploitability_delta_vs_reference.png` | Paired deltas versus the reference arm. |

Lower exploitability, lower policy-value error, and lower normalised
exploitability AUC are better. Policy loss and entropy are diagnostics, not
independent evidence of Nash convergence.
