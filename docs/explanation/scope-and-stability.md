# Scope and stability

qmatic is an early-stage library. This page states plainly what it supports,
what its assumptions are, and where the numbers can be trusted.

## What is supported

| Capability | Status |
| ---------- | ------ |
| Transient M/M/1 queue-length distribution (`mm1_len`) | Supported, with the numerical caveat below |
| Sampling a piecewise-constant rate path (`sample_mm1_len_seq`) | Experimental |
| Lindley recursion (`lindley`) | Supported |
| Mean value analysis of **closed** M/M/1 networks (`mean_value_analysis`) | Supported |
| Visit ratios of a stochastic routing matrix (`visit_ratio_vector`) | Supported |

## Assumptions

- **Closed networks only** in `mean_value_analysis`. The routing matrix must be
  square and every row must sum to one. Open networks, multiserver nodes
  (M/M/c), and general service-time distributions are out of scope.
- **A single customer class.** All customers share the same routing matrix and
  service characteristics.
- **Strictly positive rates.** `lam` and `mu` must be greater than zero; times
  and counts must be non-negative. Invalid input raises `ValueError`.

## Numerical stability

### Transient M/M/1 series

`mm1_len` evaluates an infinite Bessel series (truncated at 10,000 terms).
It is reliable when:

- `t` is moderate, or
- `t` is large **and** `lam`, `mu` are small.

It is **unstable** when `t` and the rates are both large, because intermediate
terms overflow before the exponentially small prefactor compensates. In that
regime use a matrix-exponential method or simulation. The unstable behaviour is
a property of direct evaluation, not a bug in the formula; see
[The transient M/M/1 distribution](mm1-transient.md) for details.

### Visit ratios

`visit_ratio_vector` obtains the null space of \(I - P^\top\) from an SVD, which
avoids the ill-conditioned determinant ratios used by naive Cramer's-rule
implementations. It still assumes an irreducible routing matrix; a reducible
chain (one that splits into disconnected communicating classes) has no unique
positive stationary vector.

## Performance

- `mean_value_analysis` is a short \(O(MK)\) loop and is fast for realistic
  populations.
- `mm1_len.rvs` scans the infinite support for each draw and is only practical
  for a **small number of samples**. Do not use it for large simulations.

## Stability of the public API

qmatic has not reached a 1.0 release. Function names and signatures may change
between minor versions, and the module layout is still being consolidated.
Pin an exact version if you depend on it today.
