# Leduc Poker Deep CFR Network-Size Ablation

This experiment tests whether the small Experiment 1 network is merely
sufficient for Leduc poker, or whether changing representational capacity alters
Deep CFR stability, exploitability, or average-policy value under the same
training protocol.

The default run is a 3-by-3 architecture grid:

- hidden depth: `2`, `4`, and `8` layers
- hidden width: `8`, `16`, and `32` units
- both the advantage networks and the average-policy network use the same
  architecture within a variant

The baseline arm is `layers2_width32`, matching the Experiment 1 architecture.
All non-architecture parameters are held fixed at the Experiment 1 values, and
each variant runs over three matched seeds by default. The three-seed default is
chosen so the full architecture grid completes inside the 48-hour Batch runtime
limit; five- and ten-seed seed lists remain available in `config.py` for
extended confirmation runs.

## Run

```bash
python -m experiments.leduc_poker.deep_cfr_network_size_ablation.run
```

Quick smoke run:

```bash
python -m experiments.leduc_poker.deep_cfr_network_size_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --depths 2,4 \
  --widths 8 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests
```

Useful overrides:

- `--depths`: comma-separated hidden-layer counts
- `--widths`: comma-separated hidden-layer widths
- `--variant-ids`: comma-separated subset such as `layers2_width32,layers4_width32`
- `--baseline-variant-id`: baseline used for paired deltas
- `--save-final-checkpoints`: save final full solver checkpoints

## Outputs

Each run writes to:

```text
outputs/leduc_poker_deep_cfr_network_size_ablation_<timestamp>/
```

Core thesis artifacts:

- `seed_summary.csv`: one row per `(architecture, seed)`
- `checkpoint_curves.csv`: checkpoint-level metrics and diagnostics
- `aggregate_summary.json`: per-architecture aggregate statistics
- `architecture_grid_summary.csv`: compact tabular architecture comparison
- `paired_differences_vs_baseline.csv`: seed-paired deltas against `layers2_width32`
- `paired_difference_summary.json`: aggregate paired-difference statistics
- `ablation_curves.npz`: NumPy arrays for replotting curves
- `experiment_metadata.json`: full config and software metadata

Figures:

- `exploitability_by_iteration.png`
- `exploitability_by_nodes.png`
- `average_policy_value_by_iteration.png`
- `average_policy_value_by_nodes.png`
- `policy_value_error_by_iteration.png`
- `final_exploitability_by_architecture.png`
- `final_policy_value_error_by_architecture.png`
- `exploitability_auc_by_architecture.png`
- `wall_clock_by_architecture.png`
- `paired_deltas_vs_baseline.png`
- `advantage_target_variance_diagnostic.png`
- `policy_loss_diagnostic.png`
- `policy_entropy_diagnostic.png`

## Interpretation

Exploitability remains the primary strategic-quality measure. The heatmaps are
intended as the main thesis-facing comparison because they show the depth-width
interaction without using nine separate figures. Policy-value error, policy
loss, entropy, and target variance should be treated as mechanism diagnostics:
they can explain why an architecture behaves differently, but they are not
substitutes for exploitability.
