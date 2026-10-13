# qmatic

Theoretical calculations for different types of queueing models.

## Install

```bash
pip install -e ".[dev]"
```

Runtime dependencies (`numpy`, `scipy`) are declared in `pyproject.toml`.

## Modules

- `qmatic.dists` — transient M/M/1 queue-length distribution (`mm1_len`) and
  `sample_mm1_len_seq`.
- `qmatic.lindley` — the Lindley recursion for waiting times / queue lengths.
- `qmatic.mva` — mean value analysis for closed queueing networks
  (`mean_value_analysis`, `visit_ratio_vector`, `matrix_minor`). Import explicitly:
  `from qmatic.mva import mean_value_analysis`.

## Development

```bash
python -m pytest          # tests + docstring examples
ruff check .              # lint
ruff format .             # format
```

See `ROADMAP.md` for the plan toward a mature, released package.
