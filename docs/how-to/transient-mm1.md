# Compute transient M/M/1 queue-length probabilities

This guide shows how to evaluate the transient distribution of the number of
customers in an M/M/1 queue with `mm1_len`. For the theory, see
[The transient M/M/1 distribution](../explanation/mm1-transient.md).

## Call the distribution

Create one instance and pass the four parameters positionally or by keyword, in
the order `lam, mu, i, t`:

```python
import numpy as np
from qmatic.dists import mm1_len

dist = mm1_len()

# Probability of exactly 3 customers at t=1, starting empty.
dist.pmf(3, 1.0, 2.0, 0, 1.0)
dist.pmf(3, lam=1.0, mu=2.0, i=0, t=1.0)  # equivalent
```

`pmf` accepts scalars or NumPy arrays for `k` and returns an array of the same
shape:

```python
dist.pmf(np.array([0, 1, 2]), 1.0, 2.0, 0, 1.0)
```

```
array([0.63379537, 0.25728183, 0.08240073])
```

## Verify the distribution

The probabilities over the full support sum to one. Truncate at a value large
enough that the tail is negligible:

```python
values = [dist.pmf(k, 1.0, 2.0, 0, 1.0) for k in range(200)]
sum(values)
```

```
1.0000000000000004
```

## Compute the mean

Because the state space is unbounded, compute the mean by summing `k * p_k`
over a sufficiently wide range:

```python
k = np.arange(200)
np.sum(k * dist.pmf(k, 1.0, 2.0, 0, 1.0))
```

```
0.5081243387647449
```

## Reach the steady state

For a stable queue (\(\lambda < \mu\)) the distribution converges to the
geometric steady state \((1-\rho)\rho^k\) with \(\rho = \lambda/\mu\). The
Bessel series is numerically delicate for large \(t\), so prefer the stable
regime — large `t` with small rates:

```python
lam, mu = 0.001, 0.002
rho = lam / mu
float(dist.pmf(0, lam, mu, 0, 100_000.0))  # ~ 1 - rho
```

```
0.5000000000318902
```

## Draw samples

`rvs` draws from the same distribution, but it scans the infinite support from
zero upward and is only practical for a handful of draws:

```python
dist.rvs(1.0, 2.0, 0, 1.0, size=5)
```

For any substantial simulation, use an event-driven or matrix-based approach
instead of `rvs`.

## Pitfalls

- **Parameter order.** The signature is `pmf(k, lam, mu, i, t)` — `k` comes
  first, and the shape parameters are `(lam, mu, i, t)`. Mixing them up
  silently produces wrong numbers.
- **Unstable queues.** When \(\lambda \ge \mu\) the queue has no steady state
  and the series is not meaningful.
- **Large `t` with large rates.** The infinite Bessel series loses precision
  because intermediate terms overflow before the exponentially small prefactor
  brings them back. See
  [Scope and stability](../explanation/scope-and-stability.md).
