# Leduc Poker Deep CFR Composite Learning-Rate HP Ablation

This experiment tests constant learning-rate magnitude on the current best
baseline. It deliberately does not revisit learning-rate schedules: the
earlier schedule ablation did not provide evidence that decay schedules helped
the old baseline. Here the question is narrower: whether a slightly different
constant step size is better once architecture, target processing, and
average-strategy weighting have improved.

| Variant | Change from baseline | Rationale |
| --- | --- | --- |
| `composite_best_baseline` | `0.003` | Current best constant learning-rate reference. |
| `learning_rate_0_001` | `0.001` | Re-tests the lower value from the earlier constrained search in isolation. |
| `learning_rate_0_0015` | `0.0015` | Tests a moderate stabilising reduction. |
| `learning_rate_0_002` | `0.002` | Tests a near-baseline stabilising reduction. |
| `learning_rate_0_004` | `0.004` | Tests whether faster fitting helps the stronger network. |

## Run

```bash
python -m experiments.leduc_poker.deep_cfr_composite_learning_rate_ablation.run
```

Quick smoke test:

```bash
python -m experiments.leduc_poker.deep_cfr_composite_learning_rate_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids composite_best_baseline,learning_rate_0_002 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --advantage-network-layers 8,8 \
  --batch-size-advantage 2 \
  --batch-size-strategy 2 \
  --memory-capacity 256 \
  --output-root outputs/smoke_tests
```

GCP Batch smoke test:

Set `PROJECT_ID`, `REGION`, `BUCKET`, and `SA_EMAIL` first; see
[`docs/GCP_BATCH_EXPERIMENTS.md`](../../../docs/GCP_BATCH_EXPERIMENTS.md).

```bash
./gcp/submit_batch_experiment.sh \
  "smoke-exp24-learning-rate-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.deep_cfr_composite_learning_rate_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --evaluation-interval 1 \
    --policy-network-train-every 1 \
    --variant-ids composite_best_baseline,learning_rate_0_002 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --policy-network-layers 8,8 \
    --advantage-network-layers 8,8 \
    --batch-size-advantage 2 \
    --batch-size-strategy 2 \
    --memory-capacity 256 \
    --output-root outputs/cloud/smoke-exp24-learning-rate" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

## Outputs

The runner writes the standard experiment artefacts and paired differences
against `composite_best_baseline`. For thesis figures, prefer the
`*_by_nodes.png` longitudinal charts.
