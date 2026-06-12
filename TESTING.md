# Testing and Smoke Tests

This document describes the unit-test suite and the operational smoke tests
shipped with the repository.

## 1. Environment setup

From the repository root:

```bash
python -m venv venv
source venv/bin/activate            # macOS/Linux
# .\venv\Scripts\Activate.ps1       # Windows PowerShell
pip install --upgrade pip
pip install -r requirements-dev.txt
```

`requirements-dev.txt` installs the runtime requirements plus pytest and ruff.

OpenSpiel can require platform-specific installation steps. If
`pip install -r requirements.txt` fails on `open_spiel`, follow the official
OpenSpiel install guide and rerun.

## 2. Syntax / import check

```bash
python -m compileall deep_cfr_poker experiments tests
```

This catches syntax errors. It does not catch import-resolution errors such as
missing imports, so always follow up with the unit tests below.

## 3. Unit tests

```bash
pytest
```

The suite under `tests/` covers:

- `test_replay.py` — reservoir buffer capacity invariant, Vitter Algorithm R
  uniformity over many trials, sample-without-replacement, state-dict round
  trip, and seeded determinism.
- `test_seeding.py` — `set_seed` seeds Python / NumPy / PyTorch RNGs, sets
  `PYTHONHASHSEED`, and pins cuDNN deterministic mode.
- `test_solver.py` — regret-matching uniform fallback when all advantages are
  zero, single-positive-advantage concentration, `action_probabilities` mass
  invariant, end-to-end short `solve()` returning a `SolveResult` dataclass,
  full-checkpoint round trip, and metadata-mismatch rejection on load.
- `test_experiment_utils.py` — JSON-safe coercion, threshold-crossing helpers,
  final-window mean, ragged-shape padding, NaN-aware aggregation.
- `test_snapshots.py` — filename parsing for package snapshot formats,
  snapshot directory walking, save/load round trip, and legacy version-1
  payload compatibility.
- `test_evaluation.py` — head-to-head matrix aggregation, monotonicity
  classification (clear-win / tie / clear-loss), per-checkpoint strength
  columns, across-seed strength aggregation.
- `test_smoke.py` — end-to-end two-seed mini run for experiment 1, asserts
  every expected output artefact is written.
- `test_checkpoint_head_to_head_smoke.py` — end-to-end two-seed,
  two-milestone train+analyse run for experiment 2, asserts every expected
  CSV and PNG is written and that a re-run analysis is idempotent.

Run only the fast subset:

```bash
pytest -m "not smoke"
```

Run only the smoke test:

```bash
pytest -m smoke
```

## 4. Quick experiment smoke tests

Run a small two-seed, 100-iteration version of experiment 1:

```bash
python -m experiments.leduc_poker.deep_cfr_multiseed_validation.run \
  --seeds 1234,2025 \
  --iterations 100 \
  --output-root outputs/smoke_tests
```

Run a two-seed, abbreviated-schedule version of experiment 2:

```bash
python -m experiments.leduc_poker.deep_cfr_checkpoint_head_to_head.run \
  --seeds 1234,2025 \
  --checkpoint-schedule 100,300,500 \
  --output-root outputs/smoke_tests
```

These runs are intentionally tiny; results should be used only to verify that
the code executes, not as evidence about Deep CFR performance.

## 5. Full experiment run

The default full experiment runs 10 seeds for 1,500 iterations:

```bash
python -m experiments.leduc_poker.deep_cfr_multiseed_validation.run
```

If a single seed errors mid-run, the failure is recorded in
`outputs/<run>/failed_seeds.json` and the remaining seeds are exported.

## 6. Expected testing workflow

1. `python -m compileall deep_cfr_poker experiments tests`
2. `pytest -m "not smoke"` — fast unit tests.
3. `pytest -m smoke` — end-to-end smoke test (slower; requires OpenSpiel +
   PyTorch).
4. The CLI smoke run from §4.
5. The full experiment from §5 only after §1–§4 succeed.

## 7. Linting

```bash
ruff check deep_cfr_poker experiments tests
```
