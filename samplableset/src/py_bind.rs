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

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::types::{PyAny, PyList, PyTuple};
use pyo3::{IntoPyObjectExt, prelude::*};
use std::fmt;

use crate::samplable_set::{SSetError, SamplableSet};

impl<K: fmt::Debug> From<SSetError<K>> for PyErr {
    fn from(err: SSetError<K>) -> Self {
        match err {
            SSetError::EmptySet => PyErr::new::<PyValueError, _>("The set is empty."),
            SSetError::InconsistentState(msg) => {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg)
            }
            SSetError::KeyNotFound(key) => PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Key not found in the set: {:?}",
                key
            )),
            SSetError::WeightOutOfRange { w, min, max } => {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Weight {} is out of range [{}, {}]",
                    w, min, max
                ))
            }
        }
    }
}

// Declare all supported types in one pass
macro_rules! sset_variants {
    ( $( ($ty:ty, $Variant:ident, $kind_str:literal) ),+ $(,)?) => {

        enum Inner {
            $( $Variant(SamplableSet<$ty>), )+
        }

        impl Inner {
            fn new(kind: &str, w_min: f64, w_max: f64) -> PyResult<Self> {
                match kind {
                    // TODO replace expect with proper error variant
                    $( $kind_str => Ok(
                        Inner::$Variant(
                            SamplableSet::<$ty>
                                ::new(w_min, w_max)
                                .map_err::<PyErr, _>(Into::into)?
                        )), )+
                    _ => Err(PyErr::new::<PyValueError, _>(format!("Unsupported kind: {kind}, expected one of: {}",
                                                       [$( $kind_str ),+].join(", ")))),
                }
            }

            fn size(&self) -> usize {
                match self { $( Inner::$Variant(s) => s.size(), )+ }
            }

            fn empty(&self) -> bool {
                match self { $( Inner::$Variant(s) => s.empty(), )+ }
            }

            fn exists(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
                Ok(match self {
                    $( Inner::$Variant(s) => s.exists(&key.extract::<$ty>()?), )+
                })
            }

            fn total_weight(&self) -> f64 {
                match self { $( Inner::$Variant(s) => s.total_weight(), )+ }
            }

            fn get_weight(&self, key: &Bound<'_, PyAny>) -> PyResult<f64> {
                match self {
                    $( Inner::$Variant(s) => {
                        let k: $ty = key.extract()?;
                        if !s.exists(&k) {
                            return Err(PyErr::new::<PyKeyError, _>(format!("Key not found: {:?}", k)));
                        } else {
                            s.get_weight(&k).map_err(Into::into)
                        }
                    } )+
                }
            }

            fn insert(&mut self, key: &Bound<'_, PyAny>, weight: f64) -> PyResult<bool> {
                match self {
                    $( Inner::$Variant(s) => {
                        let k: $ty = key.extract()?;
                        s.insert(&k, weight).map_err(Into::into)
                    } )+
                }
            }

            fn set_weight(&mut self, key: &Bound<'_, PyAny>, weight: f64) -> PyResult<()> {
                match self {
                    $( Inner::$Variant(s) => {
                        let k: $ty = key.extract()?;
                        s.set_weight(&k, weight).map_err(Into::into)
                    } )+
                }
            }

            fn erase(&mut self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
                match self {
                    $( Inner::$Variant(s) => {
                        let k: $ty = key.extract()?;
                        Ok(s.erase(&k))
                    } )+
                }
            }

            fn seed(&mut self, seed: u64) {
                match self {
                    $( Inner::$Variant(s) => {
                        s.seed(seed);
                    } )+
                }
            }

            fn sample<'py>(&mut self, py: Python<'py>) -> PyResult<(PyObject, f64)> {
                match self {
                    $( Inner::$Variant(s) => {
                        match s.sample()? {
                            (k, w) => {
                                let obj: PyObject = k.into_pyobject(py)
                                    .map_err(|e| PyErr::new::<PyValueError, _>(
                                        format!("Failed to convert key to PyObject: {}", e)))?.unbind().into();
                                Ok((obj, w))
                            },
                        }
                    } ),+
                }
            }

            // fn sample_ext_rng<'py>(&mut self, py: Python<'py>) -> PyResult<(PyObject, f64)> {
            //     match self {
            //         // $( Inner::$Variant(s) => s.sample_ext_rng().map(|(k, w)| (k.into_pyobject(py), w)) )+
            //     }
            // }

            fn clear(&mut self) {
                match self { $( Inner::$Variant(s) => s.clear(), )+ }
            }

            // deterministic iterator -> python list
            fn snapshot_items<'py>(&self, py: Python<'py>) -> PyResult<Vec<(PyObject, f64)>> {
                match self {
                    $( Inner::$Variant(s) => {
                        (&*s)
                            .into_iter()
                            .map(|(k, w)| {
                                // k is &T; clone/copy then convert
                                let obj: PyObject = k.clone().into_pyobject(py)?.unbind().into();
                                Ok::<(PyObject, f64), PyErr>((obj, w))
                            })
                            .collect::<PyResult<Vec<_>>>()
                    } ),+
                }
            }

            // TODO find a way to create these types without needing to pass
            // an explicit type via a string in Python
            #[inline]
            fn kind_str(&self) -> &'static str {
                match self { $( Inner::$Variant(_) => $kind_str, )+ }
            }
        }
    };
}

// Variants:
// Int
// (Int, Int)
// (Int, Int, Int)
// String
// (String, String)
// (String, String, String)
sset_variants! {
    (u64, U64, "u64"),
    ((u64, u64), Tuple2Int, "tuple2int"),
    ((u64, u64, u64), Tuple3Int, "tuple3int"),
    (String, Str, "str"),
    ((String, String), Tuple2Str, "tuple2str"),
    ((String, String, String), Tuple3Str, "tuple3str"),
}

#[pyclass(
    // module = "samplableset_rs",
    name = "SamplableSet",
    // There is really no reason to send a SamplableSet across threads
    // at this time (and it would also break the `shared_rng` guarantee, 
    // which only applies per thread). 
    // This would require refactoring NodeRef and WeakNodeRef
    // to use something like Arc<parking_lot::Mutex<T>> or Arc<std::sync::RwLock<T>>
    // as well as a refactor of the static RNG logic.
    // A potential solution to the RNG logic could involve creating a singleton 
    // instance using something like OnceLock.
    // unsendable,
)]
struct PySamplableSet {
    inner: Inner,
}

#[pymethods]
impl PySamplableSet {
    #[new]
    pub fn new(kind: &str, w_min: f64, w_max: f64) -> PyResult<Self> {
        let inner = Inner::new(kind, w_min, w_max)?;
        Ok(PySamplableSet { inner })
    }

    #[pyo3(name = "size")]
    fn py_size(&self) -> usize {
        self.inner.size()
    }

    #[pyo3(name = "empty")]
    fn py_empty(&self) -> bool {
        self.inner.empty()
    }

    #[pyo3(name = "exists")]
    fn py_exists(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.exists(key)
    }

    #[pyo3(name = "total_weight")]
    fn py_total_weight(&self) -> f64 {
        self.inner.total_weight()
    }

    #[pyo3(name = "get_weight")]
    fn py_get_weight(&self, key: &Bound<'_, PyAny>) -> PyResult<f64> {
        self.inner.get_weight(key)
    }

    #[pyo3(name = "insert")]
    fn py_insert(&mut self, key: &Bound<'_, PyAny>, weight: f64) -> PyResult<bool> {
        self.inner.insert(key, weight)
    }

    #[pyo3(name = "set_weight")]
    fn py_set_weight(&mut self, key: &Bound<'_, PyAny>, weight: f64) -> PyResult<()> {
        self.inner.set_weight(key, weight)
    }

    #[pyo3(name = "erase")]
    fn py_erase(&mut self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.erase(key)
    }

    #[pyo3(name = "seed")]
    fn py_seed(&mut self, seed: u64) {
        self.inner.seed(seed)
    }

    #[pyo3(name = "sample")]
    fn py_sample<'py>(&mut self, py: Python<'py>) -> PyResult<(PyObject, f64)> {
        self.inner.sample(py)
    }

    #[pyo3(name = "clear")]
    fn py_clear(&mut self) {
        self.inner.clear()
    }

    // TODO implement PySeqSamplableIter? This relies heavily on Python
    fn __iter__(slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let items = slf.inner.snapshot_items(py)?;

        let rows: Vec<_> = items
            .iter()
            .map(|(k, w)| PyTuple::new(py, &[k.clone_ref(py), w.into_py_any(py)?]))
            .collect::<PyResult<Vec<_>>>()?;
        let list = PyList::new(py, &rows).unwrap();

        // let mut tuples = Vec::with_capacity(items.len());
        // for (k, w) in items.iter() {
        //     tuples.push(
        //         PyTuple::new(py, &[k.clone_ref(py), w.into_py_any(py)?])
        //             .unwrap());
        // }
        // let list = PyList::new(py, &tuples);

        let iter = list.call_method0("__iter__")?;
        Ok(iter.unbind())
    }

    fn __contains__(&self, item: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.exists(item)
    }

    fn __len__(&self) -> usize {
        self.py_size()
    }

    fn __repr__(&self) -> String {
        format!(
            "SamplableSet(kind={}, size={}, total_weight={})",
            self.inner.kind_str(),
            self.inner.size(),
            self.inner.total_weight()
        )
    }
}

pub(crate) fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySamplableSet>()?;
    Ok(())
}
