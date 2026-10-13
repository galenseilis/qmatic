# The transient M/M/1 distribution

This page explains what `qmatic.dists.mm1_len` computes and why the calculation
is numerically delicate. For usage, see
[Compute transient M/M/1 queue-length probabilities](../how-to/transient-mm1.md).

## The model

An M/M/1 queue has Poisson arrivals at rate \(\lambda\), exponentially
distributed service with rate \(\mu\), a single server, and an unbounded waiting
room. The number of customers \(N(t)\) is a birth-death process with birth rate
\(\lambda\) and death rate \(\mu\). Writing \(p_k(t) = \Pr(N(t) = k \mid N(0) = i)\),
the transient probabilities satisfy the forward equations

\[
\frac{d}{dt} p_k(t) = \lambda p_{k-1}(t) + \mu p_{k+1}(t) - (\lambda + \mu) p_k(t),
\]

with boundary modifications at \(k = 0\).

## The closed form

The transient solution — the one qmatic evaluates — can be written in terms of
modified Bessel functions of the first kind \(I_n\). With
\(\rho = \lambda / \mu\) and \(a = 2\sqrt{\lambda\mu}\),

\[
p_k(t) = e^{-(\lambda + \mu)t}\left[
\rho^{(k-i)/2} I_{k-i}(at)
+ \rho^{(k-i-1)/2} I_{k+i+1}(at)
+ (1-\rho)\,\rho^{k} \sum_{j=k+i+2}^{\infty} \rho^{-j/2} I_j(at)
\right].
\]

The first two terms are the "direct" contributions; the infinite series
accounts for probability mass that has reflected off the \(k = 0\) boundary.

!!! note "A term that is easy to get wrong"

    The series is multiplied by \((1-\rho)\rho^{k}\), i.e. it scales with the
    **queried state** \(k\). Dropping that factor — as an earlier version of
    qmatic did — produces a distribution that still looks plausible but never
    converges to the correct steady state. The test suite guards against this
    regression.

## Steady state

For a stable queue (\(\rho < 1\)) the distribution converges as \(t \to \infty\)
to the geometric law

\[
\lim_{t \to \infty} p_k(t) = (1-\rho)\rho^{k}.
\]

You can use this as an end-to-end check of the implementation: evaluate the PMF
at a large time in the stable regime and compare against \((1-\rho)\rho^k\).

## Why large values are hard

The series contains factors \(\rho^{-j/2}\) that grow without bound as \(j\)
increases, multiplied by Bessel terms \(I_j(at)\) that shrink once
\(j \gg at\). The product converges, but the intermediate factors do not: for
large \(at\) the individual Bessel values overflow floating point before the
exponentially small prefactor \(e^{-(\lambda+\mu)t}\) can cancel them.

The practical consequences:

- The evaluation is **reliable for moderate \(t\)** and moderate rates.
- It is also reliable for **large \(t\) with small rates**, where \(at\) stays
  modest even though \(t\) is large.
- It is unstable when **both \(t\) and the rates are large**. Use a
  matrix-exponential or simulation method instead in that regime.

The series is truncated after at most 10,000 terms. See
[Scope and stability](scope-and-stability.md) for the full list of limitations.
