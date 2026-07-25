# Leduc Poker Deep CFR Final Candidate Validation

This experiment compares the previous best composite Deep CFR baseline with
the cumulative final candidate selected from the targeted training-parameter
ablations.

The default budget is `1050` CFR iterations and `320` traversals per iteration.
The earlier `1500`-iteration composite runs touched approximately `21.3M`
environment nodes, so this budget targets roughly `15M` nodes touched.

| Variant | Configuration |
| --- | --- |
| `composite_best_baseline` | Previous best composite baseline: residual LayerNorm centred-advantage architecture, standardised targets, uniform advantage replay, uniform average-strategy weighting, policy extraction every 25 iterations, advantage batch 1024, replay capacity 10M, learning rate 0.003. |
| `final_candidate_policy10_advbatch2048_memory5m_lr004` | Final candidate: policy extraction every 10 iterations, advantage batch 2048, replay capacity 5M, learning rate 0.004, with the same architecture, target processing, replay sampling, and average-strategy weighting. |

The default run uses five matched seeds:

```text
1234, 2025, 31415, 27182, 16180
```

For a ten-seed final comparison, use:

```text
1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001
```

## Run

From `leduc_poker_deep_cfr/leduc-poker-deep-cfr-experiments`:

```bash
python -m experiments.leduc_poker.deep_cfr_final_candidate_validation.run
```

Quick smoke test:

```bash
python -m experiments.leduc_poker.deep_cfr_final_candidate_validation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-every 1 \
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
  "smoke-exp26-final-candidate-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.deep_cfr_final_candidate_validation.run \
    --seeds 1234 \
    --iterations 3 \
    --traversals 4 \
    --evaluation-interval 1 \
    --policy-network-train-every 1 \
    --policy-network-train-steps 1 \
    --advantage-network-train-steps 1 \
    --policy-network-layers 8,8 \
    --advantage-network-layers 8,8 \
    --batch-size-advantage 2 \
    --batch-size-strategy 2 \
    --memory-capacity 256 \
    --output-root outputs/cloud/smoke-exp26-final-candidate" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

GCP Batch five-seed default run:

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-deep-cfr-exp26-final-candidate-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.deep_cfr_final_candidate_validation.run \
    --output-root outputs/cloud/leduc-deep-cfr-exp26-final-candidate" \
  "n2-standard-4" \
  "172800" \
  "4000" \
  "16000"
```

GCP Batch ten-seed run:

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-deep-cfr-exp26-final-candidate-10seed-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.leduc_poker.deep_cfr_final_candidate_validation.run \
    --seeds 1234,2025,31415,27182,16180,4242,8675309,7,99,1001 \
    --output-root outputs/cloud/leduc-deep-cfr-exp26-final-candidate-10seed" \
  "n2-standard-4" \
  "345600" \
  "4000" \
  "16000"
```

## Outputs

The runner writes the standard experiment artefacts: `seed_summary.csv`,
`checkpoint_curves.csv`, `aggregate_summary.json`,
`paired_differences_vs_baseline.csv`, `paired_difference_summary.json`,
`ablation_curves.npz`, and PNG plots. For thesis figures, prefer the
`*_by_nodes.png` longitudinal charts.
