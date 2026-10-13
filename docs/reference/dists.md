# `qmatic.dists`

Transient distribution of the number of customers in an M/M/1 queue.

`mm1_len` subclasses [`scipy.stats.rv_discrete`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html),
so it inherits the full `rv_discrete` interface (`pmf`, `cdf`, `rvs`, `mean`,
and so on). The parameters `(lam, mu, i, t)` are passed to those methods after
`k`, in that order.

::: qmatic.dists.mm1_len
    options:
      show_source: true

::: qmatic.dists.sample_mm1_len_seq
    options:
      show_source: true
