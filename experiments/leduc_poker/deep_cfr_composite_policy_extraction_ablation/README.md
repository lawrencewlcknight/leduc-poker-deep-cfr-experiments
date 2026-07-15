# Leduc Poker Deep CFR Composite Policy-Extraction HP Ablation

This experiment starts from the current best Leduc Deep CFR baseline:
residual LayerNorm centred-advantage networks, standardised unclipped advantage
targets, uniform advantage replay, and uniform average-strategy weighting.
It then varies only the average-policy extraction settings.

| Variant | Change from baseline | Rationale |
| --- | --- | --- |
| `composite_best_baseline` | None | Matched-seed reference for the current best configuration. |
| `policy_train_every_10` | Train the average-policy network every 10 iterations | Tests whether reduced extraction lag improves the final uniform average strategy. |
| `policy_train_every_50` | Train every 50 iterations | Tests whether less frequent supervised fitting reduces churn from non-stationary strategy data. |
| `policy_steps_400` | 400 policy-gradient steps per training call | Tests whether the average-policy network is underfit at 200 steps. |
| `strategy_batch_2048` | Average-policy minibatch size 2048 | Tests whether lower policy-gradient noise improves strategy extraction. |

## Run

The default run uses three matched seeds: `1234,2025,31415`.

```bash
python -m experiments.leduc_poker.deep_cfr_composite_policy_extraction_ablation.run
```

Quick smoke test:

```bash
python -m experiments.leduc_poker.deep_cfr_composite_policy_extraction_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
  --variant-ids composite_best_baseline,policy_train_every_10 \
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
  "smoke-exp21-policy-extraction-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.deep_cfr_composite_policy_extraction_ablation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --evaluation-interval 1 \
    --policy-network-train-every 1 \
    --variant-ids composite_best_baseline,policy_train_every_10 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --policy-network-layers 8,8 \
    --advantage-network-layers 8,8 \
    --batch-size-advantage 2 \
    --batch-size-strategy 2 \
    --memory-capacity 256 \
    --output-root outputs/cloud/smoke-exp21-policy-extraction" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

## Outputs

The runner writes the standard experiment artefacts: `seed_summary.csv`,
`checkpoint_curves.csv`, `aggregate_summary.json`,
`paired_differences_vs_baseline.csv`, `paired_difference_summary.json`,
`ablation_curves.npz`, and PNG plots. For thesis figures, prefer the
`*_by_nodes.png` longitudinal charts.
