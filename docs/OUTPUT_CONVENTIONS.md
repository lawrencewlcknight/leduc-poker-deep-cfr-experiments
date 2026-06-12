# Output and naming conventions

This document is the contract every experiment in this repository must satisfy
so that thesis chapters built on top of these outputs feel like one coherent
body of work. New experiments should be added by following the patterns below
rather than by inventing parallel layouts.

## Run directory layout

Every experiment writes to a single timestamped directory:

```
outputs/<experiment_name>_<YYYYMMDD_HHMMSS>/
```

The directory is created by `deep_cfr_poker.experiment_utils.create_run_dir`.
Inside the run directory, every experiment that has a meaningful concept of
each artefact must use these exact filenames:

| Filename | When written |
| --- | --- |
| `experiment_metadata.json` | Always — config, software versions, seed list, completed seeds. |
| `experiment.log` | Always — full log mirroring stdout. |
| `failed_seeds.json` | Only when at least one seed fails. |
| `seed_summary.csv` | When the experiment produces one summary row per seed. |
| `aggregate_summary.json` | When the experiment produces across-seed mean / std / SE. |
| `checkpoint_curves.csv` | When training produces per-checkpoint curves per seed. |
| `multiseed_curves.npz` | When training produces per-checkpoint curves per seed. |
| `ablation_curves.npz` | When an ablation produces per-arm, per-seed checkpoint curves. |

Other CSV / JSON / NPZ / PNG files use **snake_case topic names** without
embedded experiment names. The directory's own name encodes the experiment.
Ablation experiments should include the varied configuration axis as a column
in `seed_summary.csv` and `checkpoint_curves.csv`, rather than embedding it in
parallel filename schemes.
Paired two-arm ablations should write per-seed deltas as
`paired_differences_*.csv` and aggregate those deltas in a matching
`paired_difference_summary.json` when the comparison is meaningful.
Warm-start or resume experiments should save full checkpoints under
`checkpoints/` using the canonical `seed_<seed>_iter_<iter>_full.pt` naming and
should report paired `warm_start - baseline` differences when a continuous
baseline is part of the design.

## File-naming rules

* `topic.csv`, `topic.png`, `topic.npz` — never `experiment_topic.csv`.
* Per-(seed, iteration) artefacts live in subdirectories:
  * `checkpoints/seed_<seed>_iter_<iter>_full.pt`
  * `snapshots/seed_<seed>_iter_<iter>_snapshot.pt`
* Plot files always use `.png`. PDF conversion is left to thesis tooling.
* CSV files always use UTF-8 and CRLF-free line endings (set automatically by
  `csv.DictWriter` with `newline=""`).

## CLI conventions

Every experiment exposes a CLI entry point at:

```
experiments/<game>/<experiment_name>/run.py
```

invoked as `python -m experiments.<game>.<experiment_name>.run [phase] [flags]`.
Every CLI must support, at a minimum:

* `--output-root` — root directory under which `run_dir` is created.
* `--run-dir` — opt-out: use an existing run directory instead.
* `--seeds` — comma-separated seed list.
* `--verbose` — DEBUG-level logging.

Long-running experiments that have a generate-then-evaluate shape (e.g. the
checkpoint head-to-head experiment) split into `train` / `analyse` phases and
expose a `phase` positional argument with `all` as the default.

## Solver consistency rule

Every experiment instantiates `deep_cfr_poker.solver.DeepCFRSolver` with the
same defaults unless the experiment is *explicitly* testing a different
configuration. Configuration deviations must be documented in the
experiment's `config.py` module docstring so the thesis can cite them.
If an experiment varies policy-network training cadence, keep
`evaluation_interval` fixed across arms unless the research question explicitly
requires changing measurement cadence too.
If an experiment delays average-policy training until the final checkpoint,
intermediate strategic metrics should be written as `NaN` rather than omitted,
so all arms still share the same checkpoint grid.
If an experiment varies the optimiser learning-rate schedule, record both the
schedule id and the realised scalar learning rate in `checkpoint_curves.csv`,
and report paired schedule deltas against the constant-learning-rate baseline
where the design uses matched seeds.
Staged random-search experiments should use `stage` and `config_label` columns
in both `seed_summary.csv` and `checkpoint_curves.csv`, write sampled
configurations to `search_configurations.csv`, and keep screening and
confirmation summaries as separate CSV files as well as in
`aggregate_summary.json`.
If an experiment varies advantage-target processing, keep raw targets in replay
memory, include the target-processing mode as a column, and write processed
target variance and clipping-fraction diagnostics in `checkpoint_curves.csv`.
If an experiment varies replay sampling or average-strategy weighting, include
`advantage_replay_sampling` and `average_strategy_weighting` as columns in both
summary and curve outputs, and write priority effective sample size diagnostics
when priority replay is available.

## Snapshots and checkpoints

Lightweight policy snapshots produced via `solver.save_policy_snapshot` are
the canonical input for any cross-experiment analysis. The expected filename
for an analysis-time consumer is:

```
seed_<seed>_iter_<iter>_snapshot.pt
```

Full checkpoints (`seed_<seed>_iter_<iter>_full.pt`) are reserved for resuming
training. They are not used by analysis code.

`deep_cfr_poker.snapshots.LoadedPolicy` accepts the package snapshot format
used by the experiment runners.

## Logging

Every CLI calls `deep_cfr_poker.experiment_utils.configure_run_logging` once
near the top of `main`, which writes both to `stdout` and to
`<run_dir>/experiment.log`. Modules use `logging.getLogger(__name__)` rather
than `print`.

## Adding a new experiment — checklist

1. Create `experiments/<game>/<experiment_name>/` with `__init__.py`,
   `config.py`, `run.py` (and optionally `train.py` / `analyse.py`).
2. Use the shared lifecycle helpers (`create_run_dir`,
   `configure_run_logging`, `write_experiment_metadata`, `write_failed_seeds`,
   `write_dict_rows_csv`, `write_matrix_csv`).
3. Document the experiment's research question in
   `experiments/<game>/<experiment_name>/README.md`, plus the protocol and
   the list of output files.
4. Add a smoke test under `tests/` that runs a tiny version end-to-end.
5. Add a section to the top-level `README.md`.
6. Make sure the run directory follows the layout above.

This list is enforced by review, not by code, so the thesis can rely on the
output catalogue staying stable as the experiment count grows.
