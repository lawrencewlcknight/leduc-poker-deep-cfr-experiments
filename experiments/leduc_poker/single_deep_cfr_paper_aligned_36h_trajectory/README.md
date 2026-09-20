# Paper-aligned SD-CFR 36-hour trajectory (Experiment 30)

## Research question

Does moving Experiment 29's selected uniform SD-CFR learner toward the Leduc
training prescription in the SD-CFR paper improve its 36-hour convergence and
stability across the five thesis seeds?

## Frozen design

- Seeds `1234`, `2025`, `31415`, `27182`, and `16180`.
- One `n2-standard-8` VM per seed, with all five seeds running in parallel.
- 36 active training hours per seed.
- The Experiment 29 learner is retained except for four pre-registered changes:
  1,500 traversals per player and iteration; 750 advantage updates per fitting
  session; constant learning rate 0.001; and a fresh Adam optimiser for every
  player fitting session while retaining that player's previous network
  weights.
- Batch size 2,048, five-million-row advantage reservoirs, standardised
  targets, uniform replay, and eight-layer width-32 residual LayerNorm
  centred-advantage networks remain unchanged from Experiment 29.
- No average-policy fitting, strategy replay, or exploitability calculation in
  the timed learner. These are not components of standalone SD-CFR.
- Every fitted historical advantage network is retained. Archive-prefix
  checkpoints are recorded every 30 active-training minutes, at the first
  completed iteration crossing 15 million nodes, and at the final completed
  iteration after the 36-hour boundary.
- Exact reach-weighted uniform SD-CFR is the primary output; canonical linear
  SD-CFR is evaluated from the same archive as a secondary diagnostic.

Evaluation is deferred to five parallel post-training workers. The primary
endpoint is exact uniform-SD-CFR exploitability at 36 hours.

## Cloud execution

From the repository root, with `PROJECT_ID`, `REGION`, `BUCKET`, and `SA_EMAIL`
already set:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp30-paper-sdcfr36h-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_single_deep_cfr_paper_aligned_36h_trajectory.sh smoke-local
./gcp/run_single_deep_cfr_paper_aligned_36h_trajectory.sh run
```

The remote controller runs a cloud smoke test, launches five parallel training
workers, launches five parallel deferred-evaluation workers, and aggregates
the analysis. The laptop may be disconnected after submission.

```bash
./gcp/run_single_deep_cfr_paper_aligned_36h_trajectory.sh status
./gcp/run_single_deep_cfr_paper_aligned_36h_trajectory.sh resume
```

Principal analysis artifacts:

- `analysis/exploitability_by_training_time.png`;
- `analysis/exploitability_by_nodes.png`;
- `analysis/exploitability_by_iteration.png`;
- `analysis/uniform_minus_linear_by_training_time.png`;
- `analysis/endpoint_seed_metrics.csv`;
- `analysis/endpoint_aggregate_summary.csv`;
- `analysis/trajectory_by_training_time_summary.csv`; and
- `analysis/trajectory_by_nodes_summary.csv`.
