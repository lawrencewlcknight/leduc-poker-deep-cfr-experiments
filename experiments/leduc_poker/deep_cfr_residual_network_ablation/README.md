# Leduc Poker Deep CFR Residual-Network Ablation

Experiment 12 tests whether skip connections improve optimisation stability for
deeper Deep CFR function approximators. The run compares plain MLPs against
residual MLPs at fixed width `32` and depths `2`, `4`, and `8`.

Both the policy network and advantage networks use the same architecture within
each variant. All non-architecture parameters match the Experiment 1 baseline.

```bash
python -m experiments.leduc_poker.deep_cfr_residual_network_ablation.run
```

Quick smoke run:

```bash
python -m experiments.leduc_poker.deep_cfr_residual_network_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids plain_layers2_width32,residual_layers2_width32 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests
```

Core outputs include `seed_summary.csv`, `checkpoint_curves.csv`,
`aggregate_summary.json`, `variant_summary.csv`,
`paired_differences_vs_baseline.csv`, `ablation_curves.npz`, and thesis-facing
PNG figures.

