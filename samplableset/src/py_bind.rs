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

use pyo3::prelude::*;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::types::PyAny;
use pyo3::types::PyAnyMethods; // This import brings the extract method into scope
use std::fmt;

use crate::samplable_set::{SamplableSet, SSetError};

impl<K: fmt::Debug> From<SSetError<K>> for PyErr {
    fn from(err: SSetError<K>) -> Self {        
        match err {
            SSetError::EmptySet => PyErr::new::<PyValueError, _>("The set is empty."),
            SSetError::InconsistentState(msg) => {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg)
            }
            SSetError::KeyNotFound(key) => {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Key not found in the set: {:?}", key))
            }
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

            fn sample_py<'py>(&mut self, py: Python<'py>) -> PyResult<(PyObject, f64)> {
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

// #[pyclass(
//     module = "samplableset",
//     name = "SamplableSet"
// )]
// pub struct PySamplableSet {
//     inner: Inner,
// }