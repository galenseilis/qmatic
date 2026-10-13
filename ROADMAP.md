# QMatic Roadmap

Goal: turn QMatic from an early math sketch into a mature, trustworthy Python package for
queueing-network / queueing-system theory: correct, tested, documented, installable, and
releasable.

This roadmap is grounded in the current code. Phases are ordered by dependency: correctness
before breadth, packaging before distribution, docs alongside everything.

## Progress

Completed in the first execution pass (all verified with `pytest` + `ruff` in a `uv` venv):

- Phase 0 done: deps/extras declared in `pyproject.toml`; editable install works; machine-local
  `requirements.txt` removed; Cython leftovers dropped; version sourced from `pyproject.toml`
  and surfaced via `importlib.metadata`; broken `mva`, `lindley`, and `dists` code fixed.
- Phase 1 started: `tests/` added (26 tests incl. doctests) covering `lindley`, `mva`
  (balance + customer conservation + single-queue closed form), and `dists` (point mass at
  `t=0`, PMF sums to one, steady-state limit). Also fixed a genuine math bug: `mm1_len._pmf`'s
  third Bessel term must scale with `rho**k`, without which the distribution never reached
  steady state. Input validation added across modules.
- Phase 3/5 started: `ruff` config + `ruff format`, and `.github/workflows/ci.yml` running
  ruff + pytest on Python 3.9-3.12. Wheel builds locally and installs in a clean venv.
- Phase 4 started: Diátaxis documentation under `docs/` (tutorial, how-to, reference,
  explanation) built with MkDocs Material + mkdocstrings, math via MathJax, and a GitHub Pages
  deploy workflow (`.github/workflows/docs.yml`). `mkdocs build --strict` is clean.

Still open: doctest-cited references, `rho < 1` guard, property-based tests, type checking,
pre-commit, PyPI publishing, and the scope decisions below.

## Current state (baseline)

_Historical snapshot before the first execution pass above._
- Three ad-hoc modules under `src/qmatic`: `dists.py`, `lindley.py`, `mva.py`. No test suite,
  linter, type checker, or CI.
- Runtime deps (`numpy`, `scipy`) are undeclared in `pyproject.toml`; they only appear in the
  generated `requirements.txt`, which also pins a machine-local wheel path
  (`file:///home/galen/...`) and stale Cython/twine tooling.
- `mva.mean_value_analysis` is broken relative to its own docstring: `mu[m, :]` indexes the 1-D
  `mu` vector from the documented example, `lambdas[0]` and the final rows of `L`/`W` are never
  populated, and the `P.ndim > 2` branch references an undefined `B`. `visit_ratio_vector`
  solves `v = vP` via raw determinant ratios (Cramer's rule), which is numerically unstable and
  fails on singular `P.T`.
- `dists.sample_mm1_len_seq` contains a debug `print`, relies on loop-leaked variables for its
  final draw, and uses `assert` for input validation.
- `dists.mm1_len._pmf` has no iteration cap on its Bessel-series loop, no `rho < 1` stability
  check, and no explicit discrete support (`xk`) for the `rv_discrete` subclass.
- The repo cannot be imported/run as-is in a clean environment (no declared, installed deps).

## Guiding principles

- Correctness first: every published function should be verifiable against closed-form results,
  simulation, or reference implementations.
- Explicit scope: decide what is in (M/M/1, M/M/c, M/G/1, Jackson networks, ...) and mark the rest
  as experimental.
- Reproducible by default: `pip install -e .[dev]` gives a working environment on any machine.
- Small, reviewable changes with tests, not a big-bang rewrite.

---

## Phase 0 — Runnable and honest (unblock everything)

Make the package installable and stop the code from lying about what it does.

- [ ] Declare runtime deps (`numpy`, `scipy`) and `[project.optional-dependencies]` (`dev`) in
      `pyproject.toml`; stop deriving deps from `requirements.txt`.
- [ ] Add an editable-install path and verify `python -c "import qmatic"` works in a fresh venv.
- [ ] Remove the machine-local `QMatic @ file:///...` pin and the Cython leftovers
      (`[tool.cythonize]`, `Cython` in `requirements.txt`) unless actively reviving Cython.
- [ ] Fix or clearly quarantine broken code: repair `mean_value_analysis` for the documented 2-D
      case and delete/label the `P.ndim > 2` branch; replace the debug `print`; replace `assert`
      with real exceptions.
- [ ] Add `CHANGELOG.md` and pick a single source of truth for the version.

Acceptance: fresh clone + `pip install -e ".[dev]"` + `import qmatic` succeeds; no function's
documented example raises.

## Phase 1 — Correctness foundation

Establish trust in the math before adding models.

- [ ] Add `pytest` and a `tests/` tree mirroring module names.
- [ ] Verify each existing result against independent oracles:
      - `lindley` against a brute-force recurrence and known sample paths.
      - `mm1_len._pmf` against M/M/1 transient analytical/steady-state limits
        (`p_k -> (1-rho) rho^k`) and Monte Carlo simulation.
      - `visit_ratio_vector` against direct linear solves of `v(I-P)=0`.
      - `mean_value_analysis` against textbook MVA worked examples.
- [ ] Numerical-hardening: cap iterative loops, replace determinant-ratio solving with a
      linear solve / null-space computation, add `rho < 1` stability checks, and document
      tolerances.
- [ ] Add input validation with clear errors (positive rates, monotonic time, matching array
      lengths) and property-based tests (Hypothesis).
- [ ] Cite the derivation or reference for each formula in the docstring.

Acceptance: >= 90% line coverage on `src/qmatic`; every closed-form result matches within a
documented tolerance.

## Phase 2 — API and scope

Define a coherent, discoverable surface instead of loose functions.

- [ ] Decide the model taxonomy and module layout, e.g.:
      `models/mm1.py`, `models/mmc.py`, `models/mg1.py`, `networks/jackson.py`,
      `networks/mva.py`, `transforms/lindley.py`, `distributions/`.
- [ ] Standardize signatures (rates as `lam`/`mu`, array shapes documented, keyed parameters)
      and return types (named tuples / result objects rather than positional tuples).
- [ ] Distinguish transient vs steady-state APIs explicitly.
- [ ] Decide the simulation story (analytic only vs. a `simulate` subpackage).
- [ ] Curate `__init__.py` exports and add `__all__`; provide a stable public API.

Acceptance: a documented public API with deprecation policy; `mva` importable per the chosen
convention (currently it is intentionally not re-exported).

## Phase 3 — Packaging and release

- [ ] Adopt semantic versioning and tag releases from `main`.
- [ ] Migrate build config to a maintained backend (setuptools is fine; hatchling an option) and
      fix the package name mismatch (`QMatic` distribution vs `qmatic` import).
- [ ] Build sdist + wheel in CI, smoke-test the wheel in a clean env.
- [ ] Publish to PyPI via GitHub Actions trusted publishing (replace manual `twine.sh`).
- [ ] Add GitHub Actions CI: matrix over supported Python versions (declare `requires-python`
      honestly), running lint, type check, and tests.

Acceptance: a tagged commit produces a PyPI release automatically and installably.

## Phase 4 — Documentation and examples

- [ ] Rewrite `README.md` with install, quickstart, supported models, and a link to docs.
- [ ] Generate API docs (mkdocs + mkdocstrings or Sphinx) from NumPy-style docstrings; wire up a
      doctest runner.
- [ ] Add runnable `examples/` notebooks (transient M/M/1, Lindley recursion, closed-network
      MVA) with expected outputs.

Acceptance: new users can install and solve a queueing problem from the README alone.

## Phase 5 — Quality gates and performance

- [ ] `ruff` (lint + format), `mypy` (or pyright), and `pre-commit`; enforce in CI.
- [ ] Coverage threshold as a CI gate.
- [ ] Vectorize / cache hot paths; benchmark Bessel-series and MVA loops; keep numerical
      stability tests.
- [ ] Optional: JIT (Numba) or Cython only if profiling justifies it (Cython was removed before;
      do not reintroduce casually).

## Phase 6 — Sustainability

- [ ] `CONTRIBUTING.md`, issue/PR templates, and a `CODE_OF_CONDUCT.md`.
- [ ] Document supported Python versions and a deprecation policy.
- [ ] Reference list (textbooks/papers) for the implemented formulas.

---

## Cross-cutting decisions to make

1. Scope: analytic-only, or also a simulation engine?
2. Audience: research correctness vs. production performance/library API stability.
3. Distribution: public PyPI package vs. internal tool.
4. Target Python versions and whether to keep zero-dependency-at-import (currently requires
   numpy/scipy).

## Near-term milestone (next PR-sized chunk)

1. Declare deps + editable install (Phase 0).
2. Add pytest with `lindley` and fixed `mean_value_analysis` (2-D) tests (Phase 1).
3. Add CI running `pytest` + `ruff` (Phases 3/5).

This unblocks every later phase while fixing known-broken behavior.
