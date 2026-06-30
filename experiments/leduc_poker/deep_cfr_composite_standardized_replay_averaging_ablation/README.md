# Composite Standardised Replay And Averaging Ablation

This experiment is the Experiment 10 replay/average-strategy-weighting ablation
re-run on the current best candidate baseline:

- average-policy network: `mlp`, `32x32`
- advantage network: `residual_layer_norm_centered_advantage_mlp`, `8x32`
- target processing: `standardize`
- advantage replay: `uniform`
- baseline average-strategy weighting: `linear`

The default run uses five fixed seeds:

```text
1234, 2025, 31415, 27182, 16180
```

## Variants

The thesis-facing default variants are:

- `composite_std_uniform_replay_linear_avg_baseline`
- `composite_std_uniform_replay_uniform_avg`

Priority replay variants are also defined for exploratory runs, but they are
not included by default because the priority sampler is more expensive.

## Run

From `leduc_poker_deep_cfr/leduc-poker-deep-cfr-experiments`:

```bash
python -m experiments.leduc_poker.deep_cfr_composite_standardized_replay_averaging_ablation.run
```

Quick smoke test:

```bash
python -m experiments.leduc_poker.deep_cfr_composite_standardized_replay_averaging_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids composite_std_uniform_replay_linear_avg_baseline,composite_std_uniform_replay_uniform_avg \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests
```
