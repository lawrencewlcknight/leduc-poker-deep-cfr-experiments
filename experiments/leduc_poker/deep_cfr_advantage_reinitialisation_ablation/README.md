# Leduc Poker Deep CFR Advantage-Network Reinitialisation Ablation

This experiment tests the implementation choice controlled by
`reinitialize_advantage_networks`.

Deep CFR uses advantage networks to approximate instantaneous regrets during
traversal. Those advantage networks define the regret-matching policy that
generates later traversal data, so warm-starting versus resetting them is a
methodological choice that can affect both optimisation stability and strategic
performance.

## Research question

Holding seeds, CFR iterations, traversals, average-policy training,
architecture, optimiser settings, replay capacity, and evaluation checkpoints
fixed, does resetting the advantage networks before each advantage-network
training phase improve or harm the exploitability of the learned average
policy?

## Protocol

The default variants are:

| Variant | `reinitialize_advantage_networks` | Description |
| --- | --- | --- |
| `reinit_false_warm_started_advantage` | `False` | Advantage networks continue training across CFR iterations. |
| `reinit_true_reset_advantage` | `True` | Advantage networks are reset before each advantage-network training phase. |

The experiment is paired across seeds. The main paired output is
`True - False`, so a negative paired difference means reinitialisation improved
the metric for that seed.

Run from the repository root:

```bash
python -m experiments.leduc_poker.deep_cfr_advantage_reinitialisation_ablation.run
```

Quick smoke run:

```bash
python -m experiments.leduc_poker.deep_cfr_advantage_reinitialisation_ablation.run \
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

The default CLI uses the ten-seed thesis protocol. Use `--seeds` to select a
development subset.

## Outputs

Outputs are written to:

```text
outputs/leduc_poker_deep_cfr_advantage_reinitialisation_ablation_<YYYYMMDD_HHMMSS>/
```

Key artefacts:

| File | Contents |
| --- | --- |
| `experiment_metadata.json` | Full config, variants, seeds, software versions, and output references. |
| `experiment.log` | Full log mirroring stdout. |
| `seed_summary.csv` | One row per `(variant_id, seed)` run. |
| `checkpoint_curves.csv` | Per-checkpoint exploitability, value, loss, replay, entropy, and gradient diagnostics. |
| `aggregate_summary.json` | Mean, standard deviation, standard error, and finite count by variant. |
| `paired_differences_true_minus_false.csv` | Per-seed paired differences for key metrics. |
| `paired_difference_summary.json` | Aggregate paired-difference statistics. |
| `ablation_curves.npz` | Compact matrices for downstream thesis plotting. |
| `failed_seeds.json` | Present only if one or more runs fail. |

Generated PNGs:

| File | Figure |
| --- | --- |
| `exploitability_by_iteration.png` | Mean exploitability curves by reinitialisation setting. |
| `exploitability_by_nodes.png` | Mean exploitability by nodes touched. |
| `average_policy_value_by_iteration.png` | Mean average-policy value curves by reinitialisation setting. |
| `average_policy_value_by_nodes.png` | Mean average-policy value by nodes touched. |
| `policy_value_error_by_iteration.png` | Mean policy-value error from the known Leduc value. |
| `final_exploitability_by_variant.png` | Final exploitability bar chart. |
| `final_average_policy_value_by_variant.png` | Final average-policy value bar chart. |
| `best_exploitability_by_variant.png` | Best exploitability bar chart. |
| `final_window_exploitability_by_variant.png` | Final-window mean exploitability bar chart. |
| `exploitability_auc_by_variant.png` | Normalised exploitability AUC bar chart. |
| `advantage_target_variance_diagnostic.png` | Advantage-target variance diagnostic. |
| `advantage_grad_norm_player_0_diagnostic.png` | Player 0 advantage-gradient norm diagnostic. |
| `advantage_grad_norm_player_1_diagnostic.png` | Player 1 advantage-gradient norm diagnostic. |
| `policy_loss_diagnostic.png` | Average-policy supervised loss diagnostic. |
| `policy_normalized_entropy_mean_diagnostic.png` | Average-policy entropy diagnostic. |
| `paired_final_exploitability_delta_true_minus_false.png` | Paired final exploitability differences by seed. |

Exploitability is the primary strategic metric. Losses, gradient norms, entropy,
and advantage-target variance are diagnostics for interpreting why one
advantage-network training regime may be more stable than the other.
