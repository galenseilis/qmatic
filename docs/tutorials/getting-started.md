# Getting started

This tutorial walks you through the three things qmatic does, using a single
service system and a small closed network. By the end you will have computed a
transient queue-length distribution, analysed a closed queueing network, and
evaluated a Lindley recursion.

You do not need any queueing-theory background to follow along — the goal is to
show *what* the library computes and *how to call it*.

## Install

qmatic requires Python 3.9 or newer. It is not yet on PyPI, so install it from
a clone of the repository:

```bash
git clone https://github.com/galenseilis/qmatic
cd qmatic
pip install -e ".[dev]"
```

The `dev` extra adds the test and lint tools used in the later guides.

## 1. The number of customers in an M/M/1 queue

The `mm1_len` object models the number of customers in an M/M/1 queue. It has
four parameters, passed to `pmf` and `rvs` in this order:

| Parameter | Meaning |
| --------- | ------- |
| `lam` | arrival rate |
| `mu` | service rate |
| `i` | initial number of customers |
| `t` | time |

Suppose customers arrive at rate 1 per unit time, are served at rate 2 per unit
time, and the queue starts empty. What is the probability that exactly three
customers are present after one unit of time?

```python
import numpy as np
from qmatic.dists import mm1_len

dist = mm1_len()
dist.pmf(3, lam=1.0, mu=2.0, i=0, t=1.0)
```

```
0.021136849741508354
```

The distribution over all possible queue lengths sums to one, as it must:

```python
sum(dist.pmf(k, 1.0, 2.0, 0, 1.0) for k in range(40))
```

```
1.0000000000000004
```

You can compute the mean queue length from the same probabilities:

```python
k = np.arange(200)
np.sum(k * dist.pmf(k, 1.0, 2.0, 0, 1.0))
```

```
0.5081243387647449
```

Sampling is also available through `dist.rvs`, but it scans the (infinite)
support and is therefore only practical for a small number of draws.

## 2. A closed queueing network

A closed network has a fixed number of customers circulating between nodes. The
routing matrix `P` says where a customer goes after finishing service at a node.
Here two nodes form a cycle, with service rates 1 and 2, and four customers in
the system.

```python
from qmatic.mva import mean_value_analysis

P = np.array([[0.0, 1.0],
              [1.0, 0.0]])
mu = np.array([1.0, 2.0])

L, W, lambdas, v = mean_value_analysis(np.zeros(2), M=4, mu=mu, P=P)
```

The mean number of customers at each node grows with the population `m`, and
`L[m]` always sums to `m` because no customers are created or destroyed:

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

```python
L.sum(axis=1)
```

```
array([0., 1., 2., 3., 4.])
```

The system throughput rises as more customers cycle through:

```python
lambdas
```

```
array([1.33333333, 1.71428571, 1.86666667, 1.93548387])
```

## 3. The Lindley recursion

The Lindley recursion tracks how a queue grows and drains from an arrivals
sequence `A` and a departures sequence `D`:

```python
from qmatic.lindley import lindley

lindley(0.0, np.array([1, 2, 1]), np.array([0, 2, 1]))
```

```
array([0., 1., 1., 1.])
```

The returned array has one more entry than the inputs: it begins with the
initial value and then records the queue length after each time step.

## Next steps

- Pick a task from the [how-to guides](../how-to/index.md).
- Look up exact signatures in the [API reference](../reference/index.md).
- Understand the maths in the [explanations](../explanation/mm1-transient.md).
