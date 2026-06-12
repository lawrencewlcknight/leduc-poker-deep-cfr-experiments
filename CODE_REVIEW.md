# Code Review — Leduc Poker Deep CFR

**Reviewer:** Claude (automated review)
**Scope:** every file under `deep_cfr_poker/` and `experiments/`, plus `requirements.txt`, `README.md`, `TESTING.md`.
**Goal:** identify correctness bugs and quality issues that block "production quality".

> **Status (2026-05):** every finding below has been addressed in the
> follow-up commit. See the "Resolution status" section at the bottom for the
> per-finding mapping to the change that fixed it.

Findings are grouped by severity. Every item references file and line numbers. Suggested fixes are included where the change is small.

---

## Critical

### C1. `nn` is used but never imported in `solver.py` (instantiation will crash)

`deep_cfr_poker/solver.py` lines **103, 104, 116** reference `nn.Softmax` and `nn.MSELoss`, but the module only imports:

```python
import torch
import torch.nn.functional as F
```

`import torch.nn.functional as F` binds only `F` in the module namespace; it does **not** bind `nn`. Verified by AST inspection: no `from torch import nn` or `import torch.nn as nn` exists.

The source must import `nn` directly because `import torch.nn.functional as F` does not bind `nn` in the module namespace.

Symptom: `DeepCFRSolver(...)` raises `NameError: name 'nn' is not defined` on the first executed statement of `__init__`. Every code path that instantiates a solver — including `experiment_utils.run_single_seed` — is broken. `python -m compileall` does **not** catch this because it is a runtime name resolution error.

**Fix:** add the missing import near the other torch imports:

```python
from torch import nn
```

---

## High severity

### H1. `_traverse_game_tree` stores a meaningless `action` field in advantage memory

`solver.py` line 321:

```python
self._advantage_memories[player].add(
    AdvantageMemory(state.information_state_tensor(), int(self._iteration),
                    sampled_regret_arr, action))
```

`action` here is whatever the loop variable held on the last iteration of the preceding `for action in state.legal_actions(): ...` loop. It is not used in training (the advantage network is trained on the full per-action regret vector), and storing the last-enumerated legal action is misleading. If anyone ever tries to consume `AdvantageMemory.action` they will get a silently wrong value.

**Fix:** drop the `action` field from `AdvantageMemory` (in `replay.py`) and from the call site, or replace it with a clearly named field if it is genuinely needed (e.g., the traversing player). Since it is unused, deletion is preferable.

### H2. Default solver memory_capacity disagrees with the experiment config

`solver.py` line 49: `memory_capacity: int = int(1e6)`.
`experiments/.../config.py` line 17: `"memory_capacity": int(1e7)`.

This is not a bug in the current run path (the experiment passes its own value), but the divergence between the library default and the experiment is a smell. Combined with H3 below, defaults are inconsistent with how the code is actually run in practice.

**Fix:** pick one canonical default and keep them aligned, or make `memory_capacity` a required argument with no default.

### H3. `reinitialize_advantage_networks` default differs from experiment

`solver.py` line 53: default `True`.
`config.py` line 18: `False`.

The two together mean the published default ("Brown-style reset every iteration") is not what the experiment actually runs. Anyone reading the solver in isolation will assume reset is on. State this explicitly in the docstring and in `config.py` (`# NOTE: this disables Brown-et-al. reset; with True, advantage_network_train_steps should typically be larger`).

### H4. Silent return from training routines when batch size > buffer size

`solver.py` `_learn_advantage_network` (line 470–472) and `_learn_strategy_network` (line 507–509):

```python
if self._batch_size_advantage > len(self._advantage_memories[player]):
    return None
```

Returning `None` silently means the network is not trained at all that iteration, and the corresponding diagnostic (`policy_loss`, `*_grad_norm`) records `nan`. The experiment will continue and produce results that look fine — but the average policy will simply not have learned anything for that iteration. This can also mask configuration mistakes (e.g., `batch_size_strategy=1024` with a strategy buffer that has only collected 200 samples in early iterations).

**Fix:** at minimum log a warning the first time this happens; ideally fall back to training on the full available buffer (`min(batch_size, len(buf))`) and emit a one-time warning. For production, a hard error after, say, 10 consecutive skips would be safer.

### H5. `_advantage_target_summary` walks the entire reservoir buffer at every checkpoint

`solver.py` lines 449–463:

```python
for buffer in self._advantage_memories:
    for sample in buffer:
        values.extend(np.asarray(sample.advantage, ...).reshape(-1).tolist())
```

With `memory_capacity = 1e7` and 60 checkpoints (1500/25), this is up to ~1.2 × 10⁹ list extends across a single seed run — and Python list `.extend(.tolist())` is the slow path. For Leduc this is harmless (only ~12 infosets), but it is the kind of thing that does not scale to Leduc/HUNL and will dominate wall-clock once the buffer is non-trivial.

**Fix:** maintain online running statistics (`count`, `mean`, `M2`) on `ReservoirBuffer.add` (and a corresponding decrement on replacement), or sample a fixed-size subset (say 4096 samples) for the diagnostic.

### H6. `_policy_network_diagnostics` recurses with no depth bound

`solver.py` line 397+. Fine for Leduc (depth ≤ 3), but the function is in a "shared reusable code" package that is intended for "Leduc poker and no-limit hold'em abstractions" (per `README.md`). On larger games this will blow Python's default recursion limit (1000) and is also a quadratic-in-depth memory traversal in the worst case (`tuple(info_state.tolist())` is recreated for every visit).

**Fix:** rewrite as an explicit stack-based DFS, and (as a separate matter) cache the seen set keyed by the canonical OpenSpiel `state.information_state_string(player)` rather than tuple-ifying the full embedding tensor.

### H7. Multi-seed export is fragile to mismatched checkpoint counts

`experiment_utils.export_results` (line 270+) does:

```python
exploitability_mat = np.vstack([result["exploitability"] for result in results])
```

`np.vstack` will raise on a ragged shape. This will only happen if seeds produce different numbers of checkpoints (e.g. one seed errors out partway, or `num_iterations` differs across seeds). Any failure mid-run — including an `nn.Module` device mismatch on a single seed — currently aborts the entire run with no partial outputs. There is no `try/except` in `run.py` around `run_single_seed`.

**Fix:** wrap each seed in a try/except, persist what completed, and emit a structured error file per failed seed. Then either skip failed seeds in `export_results` or pad with NaNs.

### H8. `value_error` definition silently uses absolute value, but README says "policy-value error"

`experiment_utils.py` line 95:

```python
value_error = np.abs(avg_policy_values - LEDUC_GAME_VALUE_PLAYER_0)
```

The plot label in `plotting.py` line 93 (`r"$|v(\sigma) - (-1/18)|$"`) is consistent with this. The README phrasing is also fine, but the column in `checkpoint_curves.csv` is named `policy_value_error` — there is no signed counterpart. For a thesis-grade record you probably also want to log the **signed** error, otherwise readers cannot tell whether your average policy is over- or under-shooting the equilibrium value.

**Fix:** add a `policy_value_signed_error` (or just `policy_value`) column alongside, and decide once in the codebase what "value error" means.

---

## Medium severity

### M1. `solve()` returns a 7-element tuple

`solver.py` line 277. Positional 7-tuple returns are fragile (one new metric and every call site must change). The downstream consumer (`run_single_seed`) immediately unpacks it; if the order ever drifts, the tests will not catch it because there are no tests.

**Fix:** return a `dataclass` (`SolveResult`) with named fields, or a dict.

### M2. Tight coupling to `ReservoirBuffer` private fields

`solver.py` `extract_full_model` / `load_full_model` access `_data`, `_add_calls`, `_reservoir_buffer_capacity` directly. If `ReservoirBuffer` ever moves to e.g. a fixed-size NumPy array for performance, both methods break.

**Fix:** add `state_dict()` / `load_state_dict()` methods on `ReservoirBuffer` and use those.

### M3. `except: pass` in checkpoint code swallows real errors

`solver.py` lines 595–597 and 670–673: `try: ... except Exception: pass` around CUDA RNG state. If the serialized state genuinely is broken, the user gets silent non-determinism on resume. Catch only the specific exception (`RuntimeError` from `torch.cuda.set_rng_state_all`) and at least log a warning.

### M4. `load_full_model` does not validate against `meta`

The checkpoint stores `meta = {num_players, num_actions, embedding_size}` but `load_full_model` never reads it. Loading a checkpoint trained on `leduc_poker` into a solver constructed for `leduc_poker` will fail somewhere deep inside `load_state_dict` with a confusing tensor-shape error.

**Fix:** assert that `meta` matches the current solver up front and raise a clear error.

### M5. Inconsistent indentation across the package

`solver.py` uses **2-space** indentation (carryover from OpenSpiel/DeepMind house style). Every other Python file in the package uses **4-space**. Mixed indentation reads badly and tools like `black` will rewrite the whole file the first time they run.

**Fix:** run `black` / `ruff format` on the whole package once; pin the result with a `pyproject.toml`.

### M6. Dead/commented-out original `solve()` method

`solver.py` lines 171–196: 26 lines of commented-out code. Either delete it (preferred — the git history preserves it) or move it to `docs/` if it is actually documentation.

### M7. `policy_network_train_every` undocumented in the docstring

It is a constructor parameter, validated at line 125–126, but does not appear in the `Args:` block of the docstring (lines 56–76). A new contributor reading the public API will miss it.

### M8. Inconsistent softmax usage

`__init__` constructs `self._policy_sm = nn.Softmax(dim=-1)` (line 103), `_policy_network_diagnostics` uses it (line 419), `_learn_strategy_network` uses it (line 529), but `action_probabilities` reaches for `torch.softmax(legal_logits, dim=-1)` directly (line 384). Pick one.

### M9. `_traverse_game_tree` chance-node action conversion

`solver.py` line 301: `action = np.random.choice(chance_outcome, p=chance_proba)`. `np.random.choice` returns a numpy scalar; `state.child(action)` then receives a `numpy.int64`. OpenSpiel's pybind11 binding tolerates this for `leduc_poker`, but several games strictly require `int`. Cheap to fix:

```python
action = int(np.random.choice(chance_outcome, p=chance_proba))
```

### M10. `_traverse_game_tree` does not pass `player` to `information_state_tensor`

Line 320: `state.information_state_tensor()` (no argument) — OpenSpiel defaults to `current_player()`, which equals `player` here, so it works. But line 332 a few lines later calls `state.information_state_tensor(other_player)` explicitly. Be consistent and always pass the player; otherwise a future refactor that calls this method outside the `current_player == player` branch breaks silently.

### M11. CLI override surface is too small for a research tool

`run.py` exposes `--iterations`, `--traversals`, `--evaluation-interval`, `--learning-rate`, `--experiment-name`. It does **not** expose batch sizes, memory capacity, training steps, hidden sizes, or `--reinitialize-advantage-networks`. To explore those you have to edit `config.py`. For a thesis this is OK; for production / reproducibility you want a single CLI surface and a saved config alongside the outputs (which `experiment_metadata.json` partly does — good).

### M12. `seeding.set_seed` does not seed CUDA or `PYTHONHASHSEED`

`seeding.py`. Currently CPU-only (run.py forces `CUDA_VISIBLE_DEVICES=""`), so this is fine in practice. But the helper is documented as "Sets Python, NumPy, and PyTorch RNG seeds for reproducibility", which is misleading on a GPU host. Add `torch.cuda.manual_seed_all(seed)` (no-op on CPU) and consider `os.environ["PYTHONHASHSEED"] = str(seed)` if Python sets/dicts ever appear in deterministic paths.

### M13. `seeding.set_seed` mutates global RNG state

This is a research-code convention so I would not block on it, but the cleaner pattern is to construct a `numpy.random.Generator` and a `torch.Generator` and pass them through to the solver. That way two solvers running in the same process do not interleave their RNG draws. Worth at least a TODO.

### M14. Reservoir buffer mixes two RNGs

`replay.py`: `add` uses `np.random.randint`, `sample` uses `random.sample`. Both are seeded by `set_seed` so determinism holds, but this is a minor smell — pick one and stick with it.

### M15. `summarise_numeric_fields` raises on empty results list

`experiment_utils.py` line 170: `summary_fields = list(summary_rows[0].keys())` raises `IndexError` if `summary_rows` is empty. Combined with H7, a multi-seed run where every seed errors gives you a confusing IndexError instead of an actionable message.

### M16. `final_window_mean` window=5 is hardcoded

A 5-sample window over 60 checkpoints is a meaningful research choice. Either expose it via CLI/config, or document the choice in the README. Right now it is an unexplained magic number.

### M17. `np.array([0.] * n)` micro-pessimisation

`solver.py` line 354: `np.array([0.] * self._num_actions)` is a Python-list intermediate. Use `np.zeros(self._num_actions, dtype=np.float32)`. Also on line 316: `sampled_regret_arr = [0] * self._num_actions` is fine (it stays a list until the namedtuple is built), but if it ever moves into the hot path, prefer a NumPy array.

### M18. No tests

`TESTING.md` admits this. For thesis-quality code you can defend "no unit tests yet"; for *production* code the missing tests are the largest single gap. Specifically:

- `ReservoirBuffer`: capacity invariant, uniform-sampling property over many adds, determinism given a seed.
- `_sample_action_from_advantage`: regret-matching with all-zero advantages (uniform fallback) and with one positive advantage (deterministic).
- `action_probabilities`: legal-action mass sums to 1.
- `_traverse_game_tree`: against a hand-rolled tabular CFR on a 2-action toy game.
- `summarise_numeric_fields`: handles all-NaN columns, ddof=1 with n=1, etc.
- A 50-iteration smoke test (`pytest -k smoke`) that asserts exploitability decreased.

### M19. Logging is `print(...)`

`run.py` uses `print` for status. The output ends up in `stdout`, mixed with `tqdm`. For production switch to `logging` with a configurable level and write a structured log file alongside the run outputs.

---

## Low severity / style

### L1. `super(DeepCFRSolver, self).__init__(...)` is the Py2-style super

`solver.py` line 78. Modernise to `super().__init__(game, all_players)`.

### L2. `expected_payoff = collections.defaultdict(float)` constructed at every call

`solver.py` line 294, including for terminal and chance branches that never use it. Trivial cost on Leduc, but unidiomatic.

### L3. `action_probabilities` reshape

`solver.py` line 379: `if len(info_state_vector.shape) == 1` — `info_state_vector.ndim == 1` is more idiomatic and faster.

### L4. Inconsistent docstring style

Mix of Google-style (`Args:` / `Returns:`) and one-liner docstrings. Pick one.

### L5. `replay.py` has a bare `\` line at the very end of the file

Line 49 in the read I did showed an unterminated truncation — verify the file ends cleanly with a newline (`tail -c 5 deep_cfr_poker/replay.py`). If it does not, some toolchains (e.g. flake8 W292) will complain.

### L6. `requirements.txt` has no version pins

For reproducibility (this is the whole point of a thesis), pin versions: `torch==2.3.1`, `open_spiel==1.5`, etc. Better still, commit a `requirements.lock` produced by `pip freeze` from the environment that produced the canonical results.

### L7. `experiments/.../__init__.py` files are empty (good) but not actually files in some cases

`experiments/leduc_poker/deep_cfr_multiseed_validation/__init__.py` is empty but exists — fine. Just confirm a single trailing newline so `ls` and editors do not show "0 lines".

### L8. README claims `--traversals`, `--seeds`, etc., but no `--num-traversals`

Cross-check the README and `run.py` `argparse` definitions to make sure they do not drift.

### L9. Trailing whitespace and inconsistent blank lines in `solver.py`

Easy win once `black`/`ruff format` is in place.

### L10. `from open_spiel.python import policy` shadows the `policy` module name later used as a variable in some functions

`experiment_utils.py` line 13 (`from open_spiel.python import policy`) and line 101 (`final_policy = policy.tabular_policy_from_callable(...)`) — fine. But within `run_single_seed` you do `policy.tabular_policy_from_callable(game, solver.action_probabilities)` while `solver.solve()` also instantiates `policy.tabular_policy_from_callable(...)` internally. No bug, but the name reuse is something a fresh reader has to disambiguate.

---

## Things the code gets right

- Reservoir sampling math is correct (verified by hand for `add_calls = capacity, capacity+1, capacity+2`).
- Linear-CFR weighting (`sqrt(t)` on both sides of the MSE) is implemented correctly.
- Regret-matching fallback to uniform when cumulative regret is zero — correct and well-commented.
- Action masking at policy inference (softmax over **legal** logits only) — correct, and correctly avoids the silently-zero-mass illegal-action bug that often appears in Deep CFR implementations.
- `compute_exploitability=True` is computed on the tabular average policy via `policy.tabular_policy_from_callable`, which is the correct way to evaluate a learned function-approximated policy in OpenSpiel.
- Diagnostics record exactly the things you would want to defend in a thesis (legal-action mass before masking, normalised entropy, advantage-target variance, gradient norms per player, wall-clock).
- Output artefacts are well-structured (per-seed CSV, aggregate JSON, NPZ for replotting, metadata JSON, plus PNGs).

---

## Suggested fix order (quickest wins first)

1. **C1** — add `from torch import nn` (one line; unblocks running the package at all).
2. **H4** — turn the silent `return None` into a warning (one line; unblocks debugging).
3. **H7** — wrap each seed in a try/except in `run.py` (fewer than 10 lines; unblocks long multi-seed runs).
4. **H1** — drop `action` from `AdvantageMemory` (small, removes a footgun).
5. **M1** — convert `solve()` to a `dataclass` return.
6. **M5** — `ruff format` the package and add a `pyproject.toml`.
7. **M18** — add a minimal `tests/` directory with the five tests listed above.
8. **L6** — pin `requirements.txt`.

Everything else is incremental hardening.

---

## Resolution status

| ID | Status | Where fixed |
| --- | --- | --- |
| C1 | ✅ Fixed | `deep_cfr_poker/solver.py` — `from torch import nn` added. |
| H1 | ✅ Fixed | `replay.py` — `AdvantageMemory` is now `(info_state, iteration, advantage)`; the unused `action` field is gone. Solver no longer passes `action`. |
| H2 | ✅ Documented | `experiments/leduc_poker/deep_cfr_multiseed_validation/config.py` now has a module docstring explaining every place the experiment intentionally diverges from the solver default. |
| H3 | ✅ Documented | Same as H2: `reinitialize_advantage_networks=False` is called out in the config docstring. |
| H4 | ✅ Fixed | `solver.py` `_draw_advantage_samples` / `_draw_strategy_samples` log a one-shot WARNING and fall back to training on the full available buffer instead of silently skipping training. |
| H5 | ✅ Fixed | `_advantage_target_summary` now subsamples to at most `_ADVANTAGE_DIAGNOSTIC_MAX_SAMPLES = 4096` per checkpoint, so cost is bounded regardless of buffer capacity. |
| H6 | ✅ Fixed | `_policy_network_diagnostics` is now an iterative DFS keyed on `state.information_state_string(player)` — no recursion limit, no tuple-of-floats hashing. |
| H7 | ✅ Fixed | `experiments/.../run.py` wraps each seed in a `try/except`, logs the traceback, writes `failed_seeds.json`, and continues. `experiment_utils.export_results` and `plotting.py` right-pad ragged seeds with NaN and use `np.nanmean` / `stats.sem(nan_policy='omit')`. |
| H8 | ✅ Fixed | `experiment_utils.py` exports both `policy_value_signed_error` and `policy_value_error`; the summary row also gets `final_policy_value_signed_error`. |
| M1 | ✅ Fixed | `solver.py` defines `SolveResult` (dataclass) and `solve()` returns it. `experiment_utils.run_single_seed` consumes named attributes. |
| M2 | ✅ Fixed | `replay.py` `ReservoirBuffer` now exposes `state_dict()` / `load_state_dict()`; `extract_full_model` / `load_full_model` use them instead of poking at private fields. |
| M3 | ✅ Fixed | The two `except: pass` blocks around CUDA RNG state are now `except RuntimeError as exc:` with a `logger.warning(...)`. |
| M4 | ✅ Fixed | `load_full_model` validates `num_players`, `num_actions`, and `embedding_size` from the checkpoint `meta` against the current solver and raises a clear `ValueError` on mismatch. |
| M5 | ✅ Fixed | `solver.py` is now 4-space-indented, matching the rest of the package. A `pyproject.toml` pins `ruff` config (`line-length = 100`, `target-version = "py39"`). |
| M6 | ✅ Fixed | The 26 lines of commented-out original `solve()` are removed. |
| M7 | ✅ Fixed | `policy_network_train_every` is in the docstring's `Args:` block. |
| M8 | ✅ Fixed | `action_probabilities` now uses `self._policy_sm` consistently. |
| M9 | ✅ Fixed | `int(np.random.choice(...))` at chance nodes. |
| M10 | ✅ Fixed | `state.information_state_tensor(player)` always passes the player explicitly. |
| M11 | ✅ Fixed | `run.py` now exposes `--memory-capacity`, `--batch-size-advantage`, `--batch-size-strategy`, `--policy-network-train-steps`, `--advantage-network-train-steps`, `--reinitialize-advantage-networks`, `--compute-exploitability`, `--policy-network-layers`, `--advantage-network-layers`, `--final-window`, `--verbose`. |
| M12 | ✅ Fixed | `seeding.set_seed` now also seeds CUDA (`torch.cuda.manual_seed_all`) and `PYTHONHASHSEED`. |
| M13 | ⚠️ Documented | `seeding.py` docstring acknowledges that mutating global RNG state is the convenience choice and recommends per-experiment generators if process-level isolation is needed. |
| M14 | ✅ Fixed | `ReservoirBuffer` uses Python's `random` module exclusively for both `add` and `sample`. |
| M15 | ✅ Fixed | `summarise_numeric_fields` returns `{}` on empty input. `run.py` exits 1 with a logged error if every seed fails. |
| M16 | ✅ Fixed | `final_window` is plumbed through `run_single_seed` and exposed as the CLI flag `--final-window`. |
| M17 | ✅ Fixed | `np.zeros(self._num_actions, dtype=np.float32)` everywhere it matters. |
| M18 | ✅ Fixed | `tests/` directory: `test_replay.py`, `test_seeding.py`, `test_solver.py`, `test_experiment_utils.py`, `test_smoke.py`. The pytest suite is wired up via `pyproject.toml`. |
| M19 | ✅ Fixed | `run.py` uses the standard `logging` module, writes `experiment.log` next to the outputs, and supports `--verbose`. |
| L1 | ✅ Fixed | `super().__init__(...)` everywhere. |
| L2 | ✅ Fixed | `expected_payoff` is a plain dict, only constructed on the player branch. |
| L3 | ✅ Fixed | `info_state_vector.ndim == 1`. |
| L4 | ✅ Fixed | All public solver / utils methods use Google-style docstrings (`Args:`, `Returns:`). |
| L5 | ✅ Fixed | `replay.py` ends with a clean newline. |
| L6 | ✅ Fixed | `requirements.txt` pinned to compatible ranges; `requirements-dev.txt` added; the README explains how to regenerate a `requirements.lock` for exact reproducibility. |
| L7 | ✅ Fixed | All `__init__.py` files end with a single trailing newline. |
| L8 | ✅ Fixed | README and `TESTING.md` updated to reflect the expanded CLI surface and the new outputs (`failed_seeds.json`, `experiment.log`, signed value-error column). |
| L9 | ✅ Fixed | Whole package reformatted to consistent style. |
| L10 | ⚠️ Acknowledged | `policy` (OpenSpiel module) is still imported as a top-level name; use of `final_policy = policy.tabular_policy_from_callable(...)` is unchanged. The shadowing concern is documentation-only.

### Verification performed in this commit

- `python -m compileall deep_cfr_poker experiments tests` — clean.
- AST-level cross-reference: every `from deep_cfr_poker... import X` in the test suite resolves to a symbol defined in the imported module.
- AST-level cross-reference: every `solve_result.<field>` access in `experiment_utils.py` resolves to a field declared on `SolveResult`.
- `from torch import nn` is present in `solver.py`; the four `nn.<symbol>` references at module scope are now resolvable.

Running `pytest` end-to-end requires `torch`, `scipy`, and `open_spiel` to be installed in the user's environment; that step is intended to be run on the user's machine after `pip install -r requirements-dev.txt`.
