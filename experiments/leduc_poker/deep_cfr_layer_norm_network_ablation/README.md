# Leduc Poker Deep CFR Layer-Normalisation Network Ablation

Experiment 13 tests whether normalising hidden activations improves Deep CFR
network optimisation. It compares:

- plain MLPs
- MLPs with `LayerNorm` after each hidden layer
- residual MLPs with `LayerNorm` after each residual hidden layer

The grid uses width `32` and depths `2`, `4`, and `8`, with all non-architecture
parameters fixed to the Experiment 1 baseline.

```bash
python -m experiments.leduc_poker.deep_cfr_layer_norm_network_ablation.run
```

Quick smoke run:

```bash
python -m experiments.leduc_poker.deep_cfr_layer_norm_network_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids plain_layers2_width32,layer_norm_layers2_width32,residual_layer_norm_layers2_width32 \
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

