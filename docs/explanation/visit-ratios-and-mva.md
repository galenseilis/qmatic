# Visit ratios and mean value analysis

This page explains the ideas behind `qmatic.mva`. For usage, see
[Analyse a closed queueing network](../how-to/closed-network-mva.md).

## Closed networks and routing

In a **closed** queueing network a fixed number \(M\) of customers circulate
forever between \(K\) nodes. When a customer finishes service at node \(i\), it
moves to node \(j\) with probability \(P_{ij}\). Because customers never leave,
each row of the routing matrix \(P\) sums to one:

\[
\sum_{j=1}^{K} P_{ij} = 1.
\]

## Visit ratios

The **visit ratio** \(v_i\) is the long-run rate at which customers visit node
\(i\), measured relative to the other nodes. It satisfies the balance equation

\[
v_j = \sum_{i=1}^{K} v_i P_{ij},
\]

or in matrix form \(v = vP\). This means \(v\) is a left eigenvector of \(P\)
associated with the eigenvalue 1 — equivalently, the stationary distribution of
the routing Markov chain.

Only the *direction* of \(v\) is determined; any positive scalar multiple is a
valid solution. `qmatic.mva.visit_ratio_vector` fixes the scale by normalising
\(v\) to sum to one, and finds it from the null space of \(I - P^\top\) using a
singular value decomposition. This is far more robust than solving the system
with explicit determinants, which becomes numerically unstable as the network
grows.

The downstream mean value analysis is invariant to the scale of \(v\), so the
choice of normalisation does not change the results.

## Mean value analysis

Once \(v\) is known, the mean value analysis recursion computes, for each
population \(m = 1, 2, \dots, M\):

\[
W_i(m) = \frac{1 + L_i(m-1)}{\mu_i},
\qquad
X(m) = \frac{m}{\sum_{i=1}^{K} v_i W_i(m)},
\qquad
L_i(m) = X(m)\, v_i\, W_i(m).
\]

- \(W_i(m)\) — the mean **residence time** at node \(i\) (waiting plus service)
  under the arrival theorem: a customer arriving to a network with \(m\)
  customers sees, on average, the equilibrium of the network with \(m-1\)
  customers.
- \(X(m)\) — the **system throughput**, in customers per unit time, defined so
  that the total work \(\sum_i v_i W_i(m)\) per customer equals \(m / X(m)\).
- \(L_i(m)\) — the resulting **mean queue length** at each node.

The recursion starts from the empty network, \(L_i(0) = L_{0,i}\), and returns
the arrays `L`, `W`, `lambdas`, and `v` described in the
[reference](../reference/mva.md).

## Why conservation is a good test

Because no customers are created or destroyed,

\[
\sum_{i=1}^{K} L_i(m) = m \quad \text{for every } m.
\]

qmatic uses this invariant in its test suite: `L.sum(axis=1)` must reproduce
`0, 1, 2, ..., M`. Any change to the recursion that accidentally loses or
duplicates customers will fail this check immediately.
