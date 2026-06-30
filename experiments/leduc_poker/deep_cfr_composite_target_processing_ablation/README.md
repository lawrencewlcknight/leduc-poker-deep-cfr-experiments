# Composite Target-Processing Ablation

This experiment mirrors the Leduc Deep CFR target-processing ablation, but the
raw-target baseline is the proposed composite architecture:

- average-policy network: `mlp`, `32x32`
- advantage network: `residual_layer_norm_centered_advantage_mlp`, `8x32`
- advantage networks are warm-started across CFR iterations
- target processing is the only varied intervention

The default run uses five fixed seeds:

```text
1234, 2025, 31415, 27182, 16180
```

## Variants

- `composite_raw_targets_baseline`: composite architecture with raw advantage
  targets.
- `composite_standardized_targets`: composite architecture with batch-standardised
  advantage targets.
- `composite_clipped_targets`: composite architecture with targets clipped to
  `[-1.0, 1.0]`.
- `composite_standardized_clipped_targets`: composite architecture with
  standardisation followed by clipping.

## Run

From `leduc_poker_deep_cfr/leduc-poker-deep-cfr-experiments`:

```bash
python -m experiments.leduc_poker.deep_cfr_composite_target_processing_ablation.run
```

Quick smoke test:

```bash
python -m experiments.leduc_poker.deep_cfr_composite_target_processing_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids composite_raw_targets_baseline,composite_standardized_targets \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests
```
