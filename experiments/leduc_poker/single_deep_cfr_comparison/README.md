# Experiment 28 — paired Deep CFR versus Single Deep CFR

This experiment tests whether Single Deep CFR (SD-CFR) improves the quality of
the output strategy by eliminating Deep CFR's separately fitted average-policy
network.

Each seed runs **one** Deep CFR training trajectory using the selected
Experiment 26/27 final-candidate configuration. The same fitted advantage
networks therefore produce all three evaluation arms:

| Arm | Output strategy |
| --- | --- |
| `Deep CFR` | The conventional average-policy network fitted from uniformly weighted strategy replay. |
| `SD-CFR (uniform)` | The historical advantage-network mixture with uniform iteration weights and exact player-own-reach weighting. This is the primary, weighting-matched representation comparison. |
| `SD-CFR (linear)` | The historical advantage-network mixture with canonical linear iteration weights and exact player-own-reach weighting. This is the secondary canonical-algorithm comparison. |

All arms are paired: traversals, sampled regret targets, advantage fitting and
random seed are identical within each seed. The `Deep CFR` versus uniformly
weighted `SD-CFR` contrast isolates average-strategy representation. The
canonical linearly weighted arm also changes the averaging rule, so it must not
be interpreted as a pure representation ablation.

## Correct SD-CFR semantics

The experiment archives each player's advantage network immediately after that
player's update. For exact Leduc evaluation it reconstructs

\[
\bar\sigma_i(I,a)=
\frac{\sum_t t\,\pi_i^{\sigma^t}(I)\sigma_i^t(I,a)}
     {\sum_t t\,\pi_i^{\sigma^t}(I)}.
\]

The uniformly weighted arm replaces (t) with (1). Neither arm takes a
pointwise mean of network policies. The saved archive is also playable through
`deep_cfr_poker.sd_cfr.SampledSDCFRPolicy`, which samples one historical network
for each player at the start of a game and holds that choice fixed throughout
the trajectory. Pass `weighting="uniform"` or `weighting="linear"` to select the
deployed mixture.

## Default design

- Five matched seeds: `1234, 2025, 31415, 27182, 16180`.
- The Experiment 26-selected configuration used as the single-arm candidate in
  Experiment 27.
- 1,050 CFR iterations and 320 traversals per player per iteration
  (approximately 15 million nodes).
- Residual LayerNorm centred-advantage network with eight width-32 layers;
  two-layer width-32 policy MLP.
- Standardised advantage targets, uniform advantage replay and uniform
  average-strategy weighting.
- Average-policy fitting every 10 iterations, advantage minibatch 2,048,
  replay capacity 5 million and constant learning rate 0.004.
- Exact evaluation every 25 iterations.
- Uniform weighting-matched and canonical linear SD-CFR evaluation.
- A playable final SD-CFR archive and conventional Deep CFR policy snapshot for
  every seed.

## Local smoke test

Run from the repository root:

```bash
python -m experiments.leduc_poker.single_deep_cfr_comparison.run \
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

## Full local run

```bash
python -m experiments.leduc_poker.single_deep_cfr_comparison.run
```

## Resilient GCP Batch run

After setting `PROJECT_ID`, `REGION`, `BUCKET`, `SA_EMAIL`, and `REPO_REF`:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp28-sdcfr-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5
./gcp/run_single_deep_cfr_comparison.sh run
```

The remote controller runs a cloud smoke test, launches one `n2-standard-4` VM
per seed, and aggregates the five durable worker results. The laptop may be
disconnected immediately after the controller is submitted. Monitor with:

```bash
./gcp/run_single_deep_cfr_comparison.sh status
```

If a cloud task fails, retain the same `RUN_ID` and run
`./gcp/run_single_deep_cfr_comparison.sh resume`. Successful workers have a
`SUCCESS.json` marker and are not retrained. Each worker also writes its raw
checkpoint curve before calculating summary statistics, preventing a later
analysis error from discarding an expensive training trajectory.

## Outputs

- `seed_summary.csv`
- `checkpoint_curves.csv`
- `aggregate_summary.json`
- `paired_difference_summary.json`
- `exploitability_by_iteration.png`
- `exploitability_by_nodes.png`
- `exploitability_by_training_time.png`
- `final_exploitability_paired.png`
- `sd_cfr_archives/seed_<seed>_sd_cfr_archive.pt`
- `policy_snapshots/seed_<seed>_deep_cfr_policy.pt`
- `seed_results/seed_<seed>_checkpoint_curves.csv` and
  `seed_results/seed_<seed>_result.json` (durable per-seed recovery files)
- `experiment_metadata.json`, `experiment.log`, and optional
  `failed_seeds.json`

The CSV and JSON fields identify `uniform` and `linear` explicitly. Negative
values of `sd_cfr_<weighting>_minus_deep_cfr_*` favour that SD-CFR arm.
