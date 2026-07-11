# Leduc Poker Deep CFR Composite Advantage-Fitting HP Ablation

This experiment tests whether the deeper residual advantage networks selected
by the architecture experiments need different supervised fitting effort from
the baseline. The policy architecture, traversal budget, target processing,
replay scheme, and average-strategy weighting are fixed.

| Variant | Change from baseline | Rationale |
| --- | --- | --- |
| `composite_best_baseline` | None | Matched-seed reference for the current best configuration. |
| `advantage_steps_100` | 100 advantage steps | Tests whether 200 steps are more than required. |
| `advantage_steps_400` | 400 advantage steps | Tests whether deeper regret approximators are underfit. |
| `advantage_batch_512` | Advantage batch size 512 | Tests whether more stochastic regret fitting is beneficial. |
| `advantage_batch_2048` | Advantage batch size 2048 | Tests whether lower-noise regret gradients improve equilibrium quality. |
| `advantage_batch_2048_steps_400` | Batch size 2048 and 400 steps | Tests the most plausible high-fitting-effort combination. |

## Run

```bash
python -m experiments.leduc_poker.deep_cfr_composite_advantage_fitting_ablation.run
```

Quick smoke test:

```bash
python -m experiments.leduc_poker.deep_cfr_composite_advantage_fitting_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids composite_best_baseline,advantage_batch_512 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests
```

## Outputs

The runner writes the standard experiment artefacts and paired differences
against `composite_best_baseline`. For thesis figures, prefer the
`*_by_nodes.png` longitudinal charts.
