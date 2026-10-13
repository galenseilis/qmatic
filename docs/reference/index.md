# API reference

The pages in this section are generated directly from the source docstrings.

## Modules

| Module | Public objects |
| ------ | -------------- |
| [`qmatic.dists`](dists.md) | `mm1_len`, `sample_mm1_len_seq` |
| [`qmatic.lindley`](lindley.md) | `lindley` |
| [`qmatic.mva`](mva.md) | `matrix_minor`, `visit_ratio_vector`, `mean_value_analysis` |

## Import convention

`qmatic.dists` and `qmatic.lindley` are available from the package root.
`qmatic.mva` is not imported automatically:

```python
import qmatic
from qmatic import dists, lindley          # available
from qmatic.mva import mean_value_analysis  # import explicitly
```

## Conventions

- **Array arguments** are one-dimensional unless stated otherwise.
- **Rates** (`lam`, `mu`) are strictly positive; **times** and **counts** are
  non-negative.
- **Invalid input** raises `ValueError` with a message describing the violated
  condition.
- **Pure functions.** Results are returned, not written back into the inputs.
