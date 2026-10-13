# Analyse a closed queueing network with mean value analysis

This guide shows how to compute mean queue lengths, residence times, and
throughput for a closed network of M/M/1 queues using `mean_value_analysis`.
For the theory, see
[Visit ratios and mean value analysis](../explanation/visit-ratios-and-mva.md).

## Describe the network

You need three things:

1. **Service rates** `mu` — one positive rate per node.
2. **Routing matrix** `P` — a square matrix whose rows sum to one; `P[i, j]` is
   the probability a customer finishing at node `i` next goes to node `j`.
3. **Population** `M` — the number of customers circulating in the closed
   network.

```python
import numpy as np
from qmatic.mva import mean_value_analysis

P = np.array([[0.0, 1.0],
              [1.0, 0.0]])   # two nodes in a cycle
mu = np.array([1.0, 2.0])
L_0 = np.zeros(2)             # initial mean queue lengths
```

## Run the analysis

```python
L, W, lambdas, v = mean_value_analysis(L_0, M=4, mu=mu, P=P)
```

The four returned values are:

| Name | Shape | Meaning |
| ---- | ----- | ------- |
| `L` | `(M + 1, K)` | mean queue length at each node, by population `m` |
| `W` | `(M, K)` | mean residence time at each node, by population `m` |
| `lambdas` | `(M,)` | system throughput for populations `1..M` |
| `v` | `(K,)` | visit ratios, normalised to sum to one |

## Read the results

`L[m]` is the vector of mean queue lengths when `m` customers are in the
network, and `W[m-1]` is the corresponding residence time:

```python
L
```

```
array([[0.        , 0.        ],
       [0.66666667, 0.33333333],
       [1.42857143, 0.57142857],
       [2.26666667, 0.73333333],
       [3.16129032, 0.83870968]])
```

Because the network is closed, `L[m].sum()` must equal `m`. Use this as a
sanity check:

```python
np.testing.assert_allclose(L.sum(axis=1), np.arange(M + 1))
```

Throughput increases monotonically with the population:

```python
lambdas
```

```
array([1.33333333, 1.71428571, 1.86666667, 1.93548387])
```

## Get the visit ratios alone

If you only need the relative visit rates, call `visit_ratio_vector` directly.
It solves `v = v P` and returns a non-negative vector summing to one:

```python
from qmatic.mva import visit_ratio_vector

visit_ratio_vector(P)
```

```
array([0.5, 0.5])
```

## Pitfalls

- **Open networks are not supported.** `P` must be square and every row must
  sum to one. A substochastic matrix (some row sums below one, i.e. customer
  departures) raises `ValueError`.
- **`mu` is one-dimensional.** Pass a per-node vector such as
  `np.array([1.0, 2.0])`, not a two-dimensional array.
- **`L_0` must match the number of nodes.** It is the starting point of the
  recursion; use `np.zeros(K)` unless you have a reason not to.
- **Conservation is a cheap correctness check.** Always confirm that
  `L.sum(axis=1)` equals `np.arange(M + 1)`.
