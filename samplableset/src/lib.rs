// MIT License
//
// Copyright (c) 2025 Jai Veilleux
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Weighted sampling with composition–rejection and dyadic (power-of-two) grouping.
//!
//! This crate implements the sampler from:  
//! G. St-Onge, J.-G. Young, L. Hébert-Dufresne, L. J. Dubé,  
//! *Efficient sampling of spreading processes on complex networks using a composition and rejection algorithm*,  
//! **Comput. Phys. Commun.** 240 (2019) 30–37. DOI: [10.1016/j.cpc.2019.02.008](https://doi.org/10.1016/j.cpc.2019.02.008)
//!
//! Let $W = \dfrac{w_{\max}}{w_{\min}}$. Items are partitioned into approximately
//! $G = \lceil \log_2 W \rceil + 1$ propensity groups. A cumulative tree over groups
//! enables group selection in $\mathcal{O}(\log G) = \mathcal{O}(\log\log W)$ time,
//! and the subsequent acceptance–rejection step runs in $\mathcal{O}(1)$ expected time.
//!
//! If $W$ is bounded in your application, operations are effectively
//! $\mathcal{O}(1)$ on average.

// Only compiles the module if py_bind feature is enabled
#[cfg(feature = "py_bind")]
mod py_bind;
#[cfg(feature = "py_bind")]
use pyo3::{pymodule, types::PyModule, Bound, PyResult};

mod binary_tree;
mod hash_propensity;
pub mod samplable_set;

pub use samplable_set::SamplableSet;

#[cfg(feature = "py_bind")]
#[pymodule]
fn samplableset_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py_bind::register_py(m)?;
    
    Ok(())
}
