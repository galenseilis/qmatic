# qmatic

**qmatic** provides theoretical calculations for queueing models: the transient
M/M/1 queue-length distribution, mean value analysis for closed queueing
networks, and the Lindley recursion. It is built on
[NumPy](https://numpy.org/) and [SciPy](https://scipy.org/).

```bash
git clone https://github.com/galenseilis/qmatic
cd qmatic
pip install -e .
```

!!! note "PyPI release pending"

    qmatic is not yet published to PyPI. Install from the repository as shown
    above; a `pip install qmatic` release is planned.

!!! warning "Early-stage project"

    qmatic is under active development. Read
    [Scope and stability](explanation/scope-and-stability.md) before relying on
    any result in production.

## Where to start

These pages follow the [Diátaxis](https://diataxis.fr/) framework, which splits
documentation by what you need at a given moment.

- **Learning?** Begin with the [Getting started tutorial](tutorials/getting-started.md).
- **Solving a specific task?** See the [how-to guides](how-to/index.md).
- **Looking up a function?** Go to the [API reference](reference/index.md).
- **Want the theory behind a calculation?** Read the
  [explanations](explanation/mm1-transient.md).

## Modules at a glance

| Module | What it provides |
| ------ | ---------------- |
| [`qmatic.dists`](reference/dists.md) | Transient M/M/1 queue-length distribution and sampling |
| [`qmatic.lindley`](reference/lindley.md) | Lindley recursion for waiting times / queue lengths |
| [`qmatic.mva`](reference/mva.md) | Mean value analysis of closed queueing networks |

`qmatic.dists` and `qmatic.lindley` are re-exported from the package root;
`qmatic.mva` is intentionally **not** imported automatically and must be
imported explicitly:

```python
from qmatic.mva import mean_value_analysis
```
