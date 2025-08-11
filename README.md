# SamplableSet (Rust)

Fast weighted sampling **with replacement** using a composition–rejection scheme and **dyadic (power-of-two) grouping**. This crate is a Rust re-implementation and extension of the approach introduced by my professors
Guillaume St-Onge, Laurent Hébert-Dufresne, their collaborators, and the original
C++ project **[SamplableSet](https://github.com/gstonge/SamplableSet)**.

> **Paper**  
> G. St-Onge, J.-G. Young, L. Hébert-Dufresne, L. J. Dubé,  
> *Efficient sampling of spreading processes on complex networks using a composition and rejection algorithm*,  
> **Comput. Phys. Commun.** 240 (2019) 30-37. DOI: [10.1016/j.cpc.2019.02.008](10.1016/j.cpc.2019.02.008)

> **Original repository (C++)**  
> https://github.com/gstonge/SamplableSet

---

## Why this structure

Let $W = \dfrac{w_{\max}}{w_{\min}}$ be the weight range. Items are binned into  
$G = \lceil \log_2 W \rceil + 1$ **propensity groups** by scale. A cumulative **binary tree over
groups** enables group selection in $\mathcal{O}(\log G) = \mathcal{O}(\log\log W)$ time, and
an in-group acceptance–rejection step succeeds in 
$\mathcal{O}(1)$ expected time.

If $W$ is bounded in your application, operations are effectively **$\mathcal{O}(1)$** on average.

**Average-case complexity**

- **`sample`**: $\mathcal{O}(\log\log W) + \mathcal{O}(1)$ expected  
- **`insert` / `erase` / `set_weight`**: $\mathcal{O}(\log\log W)$ (update one group total) + $\mathcal{O}(1)$
bucket ops

---

## Status

- ✅ Core `SamplableSet<T>` (Rust)
- ✅ Deterministic iteration over stored items
- ✅ Probabilistic iteration over stored items
- ✅ Sampling with internal RNG (`sample`) and external RNG (`sample_ext_rng`)
- ✅ Basic unit tests: invariants, distribution checks, fuzz-style mutate/sample
- 🚧 Integration tests
- 🚧 Python bindings (PyO3)
- 🚧 C bindings (cbindgen)
- 🚧 Benchmarks & full docs

> **Public surface:** `SamplableSet` is the sole public entry point. Internal helpers
> (`BinaryTree`, `HashPropensity`, etc.) remain crate-private.

---

## Install (Rust)

```toml
# Cargo.toml
[dependencies]
samplable-set = { git = "https://github.com/<you>/samplable-set-rs" }
```
