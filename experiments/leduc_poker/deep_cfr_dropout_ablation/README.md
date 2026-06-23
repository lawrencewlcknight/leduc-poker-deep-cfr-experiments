# Experiment 17: Dropout Regularisation Ablation

This experiment is intended as a conservative negative/control-style test of
dropout in Deep CFR. Dropout may regularise supervised advantage fitting, but
the regression targets are already noisy and Leduc poker is small, so the
reasonable prior is that dropout may hurt more often than it helps.

The average-policy network is held fixed at the Experiment 1 two-layer,
width-32 MLP. Dropout is applied only to the advantage networks during
supervised fitting; regret-matching queries and evaluation run with dropout
disabled. The design compares dropout probabilities `0.00`, `0.05`, `0.10`,
and `0.20` at advantage-network depths `2` and `8`, both with hidden width
`32`.

```bash
python -m experiments.leduc_poker.deep_cfr_dropout_ablation.run
```

Quick local smoke test:

```bash
python -m experiments.leduc_poker.deep_cfr_dropout_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids dropout_p00_advantage_layers2_width32,dropout_p05_advantage_layers2_width32 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests
```

The runner writes the same thesis-facing artefacts as the other architecture
ablations, including per-seed summaries, checkpoint curves, variant summaries,
paired differences against the no-dropout 2x32 baseline, and diagnostic figures.
