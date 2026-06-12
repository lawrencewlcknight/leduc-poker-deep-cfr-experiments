# Leduc Poker Deep CFR Final-Only Policy-Training Ablation

This experiment tests whether Deep CFR performance changes when the
average-policy network is trained intermittently during CFR or only once after
CFR data collection has finished.

The advantage networks drive traversal and affect future regret samples. The
average-policy network approximates the average strategy represented by policy
memory, so it can be treated as an extraction model rather than a model needed
to generate the next CFR iteration. This ablation separates that methodological
choice from the core Deep CFR data-collection process.

## Research question

Holding game, seeds, CFR iterations, traversals, advantage training, replay
capacity, architecture, optimiser settings, and evaluation checkpoints fixed,
does final-only average-policy extraction match or change the quality of the
final evaluated average policy?

## Protocol

The default variants are:

| Variant | Description |
| --- | --- |
| `intermittent_every_25` | Baseline: train the average-policy network every 25 CFR iterations for 200 gradient steps. |
| `final_only_200_steps` | Train the average-policy network once at the end for 200 gradient steps. |
| `final_only_matched_steps` | Train once at the end with the same total policy-gradient-step budget as the intermittent baseline. |

For final-only arms, intermediate exploitability and policy-value entries are
recorded as missing because no trained average-policy network exists before the
final extraction. The final checkpoint is therefore the primary comparison.

Run from the repository root:

```bash
python -m experiments.leduc_poker.deep_cfr_final_only_policy_training_ablation.run
```

Quick smoke run:

```bash
python -m experiments.leduc_poker.deep_cfr_final_only_policy_training_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --variant-ids intermittent_every_25,final_only_200_steps \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 0 \
  --batch-size-strategy 0 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests
```

Use `--seeds 1234,2025,31415,27182,16180,4242,8675309,7,99,1001` for the
extended ten-seed thesis run if compute budget permits.

## Outputs

Outputs are written to:

```text
outputs/leduc_poker_deep_cfr_final_only_policy_training_ablation_<YYYYMMDD_HHMMSS>/
```

Key artefacts:

| File | Contents |
| --- | --- |
| `experiment_metadata.json` | Full config, variants, seeds, software versions, and output references. |
| `experiment.log` | Full log mirroring stdout. |
| `seed_summary.csv` | One row per `(variant_id, seed)` run. |
| `checkpoint_curves.csv` | Per-checkpoint curves, including missing intermediate strategic metrics for final-only arms. |
| `aggregate_summary.json` | Mean, standard deviation, standard error, and finite count by variant. |
| `paired_differences_vs_reference.csv` | Per-seed deltas versus `intermittent_every_25`. |
| `ablation_curves.npz` | Compact matrices for downstream thesis plotting. |
| `failed_seeds.json` | Present only if one or more runs fail. |

Generated PNGs:

| File | Figure |
| --- | --- |
| `exploitability_by_iteration.png` | Mean exploitability curves by policy-training regime. |
| `exploitability_by_nodes.png` | Mean exploitability by nodes touched. |
| `average_policy_value_by_iteration.png` | Mean average-policy value curves by policy-training regime. |
| `average_policy_value_by_nodes.png` | Mean average-policy value by nodes touched. |
| `policy_value_error_by_iteration.png` | Mean policy-value error from the known Leduc value. |
| `final_exploitability_by_regime.png` | Final exploitability bar chart. |
| `final_average_policy_value_by_regime.png` | Final average-policy value bar chart. |
| `final_policy_value_error_by_regime.png` | Final policy-value error bar chart. |
| `cumulative_policy_training_events.png` | Cumulative policy-training events by regime. |
| `cumulative_policy_gradient_steps.png` | Cumulative policy-gradient steps by regime. |
| `policy_loss_by_regime.png` | Average-policy supervised loss diagnostic. |
| `policy_entropy_by_regime.png` | Average-policy entropy diagnostic. |
| `paired_final_exploitability_delta_vs_reference.png` | Paired deltas versus the intermittent baseline. |

Lower exploitability and lower policy-value error are better. Policy loss,
entropy, and update-budget diagnostics help interpret the extraction procedure
but should not be treated as independent evidence of Nash convergence.
