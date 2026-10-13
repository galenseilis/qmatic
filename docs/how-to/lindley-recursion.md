# Track queue lengths with the Lindley recursion

The Lindley recursion turns a sequence of arrivals and departures into a queue
length over time:

\[
L_{t+1} = \max\left(0,\; L_t + A_t - D_t\right)
\]

This guide shows how to evaluate it. For context, see
[Scope and stability](../explanation/scope-and-stability.md).

## Evaluate a short trace

```python
import numpy as np
from qmatic.lindley import lindley

L = lindley(0.0, np.array([1, 2, 1]), np.array([0, 2, 1]))
L
```

```
array([0., 1., 1., 1.])
```

The result has one more entry than the inputs. `L[0]` is the starting value
`L_0`, and `L[t + 1]` is the queue length after applying the `t`-th arrival and
departure. Notice that the queue is clamped at zero whenever departures exceed
arrivals plus the current length.

## Start from a non-empty queue

`L_0` sets the initial queue length:

```python
lindley(5.0, np.array([0, 0]), np.array([2, 2]))
```

```
array([5., 3., 1.])
```

## Work with a longer trace

`A` and `D` can be any equal-length one-dimensional arrays, for example random
arrivals with fixed service:

```python
rng = np.random.default_rng(0)
arrivals = rng.integers(0, 3, size=10)
departures = np.ones(10, dtype=int)

lindley(0.0, arrivals, departures)
```

```
array([0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1.])
```

## Pitfalls

- **Arrays must be the same length.** A mismatch raises `ValueError`.
- **The result is one longer than the inputs.** Do not zip the result directly
  against `A` or `D` without dropping the first element.
- **`L_0` is not modified.** The function is pure and returns a new array.
