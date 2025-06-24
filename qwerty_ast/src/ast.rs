//! This is a proof-of-concept for a simple enum and struct based AST
//! structure, with natural nested nodes and fat pointers/Boxes for
//! holding child nodes

// FIXME: Remove later!
#![allow(unused_imports, dead_code)]
use crate::basis::*;
use crate::dimexpr::{DimExpr, DimVar, DimVarValue};
use crate::inference::{ConstraintInfo, Context, DimConstraint};
use crate::types::Type;
use crate::types::*;
use num::complex::Complex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyString, PyList};
use std::collections::{HashMap, HashSet};
pub use crate::basis::{OneQubit, ZeroQubit};
use crate::dimexpr::DimExpr::Number;

#[derive(Debug, Clone)]
pub enum ASTNode {
    ZeroQubit(ZeroQubit),
    OneQubit(OneQubit),
    Tensor(Tensor),
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct NodeBox {
    ptr: Box<ASTNode>,
}

#[pymethods]
impl NodeBox {
    #[staticmethod]
    fn new_zero_qubit(str: &PyString) -> PyResult<NodeBox> {
        let s = str.to_string();
        if s.len() != 1 {
            // TODO: Check if ASCII encoding and not some silly emoji :)
            return Err(PyValueError::new_err(format!("bruh wtf is {} ???", s)));
        }
        let c = s.as_bytes()[0];
        Ok(NodeBox {
            ptr: Box::new(ASTNode::ZeroQubit {
                0: ZeroQubit { sym: c, typ: None },
            }),
        })
    }
    #[staticmethod]
    fn new_tensor(list: &PyList) -> PyResult<NodeBox> {
        let mut nodes = vec![];
        for e in list {
            // let b = e.downcast::<NodeBox>();
            nodes.push(*e.extract::<NodeBox>()?.ptr);
        }
        Ok(NodeBox {
            ptr: Box::new(ASTNode::Tensor {
                0: Tensor {
                    elts: nodes,
                    factor: DimExpr::from(1),
                    typ: None,
                },
            }),
        })
    }

    fn get_tensor(&self) -> PyResult<Option<String>> {
        match *self.ptr {
            ASTNode::ZeroQubit(ref qs) => {
                Ok(Some((qs.sym as char).to_string()))
            }
            ASTNode::Tensor(ref tensor) => {
                let mut result = String::new();

                for node in &tensor.elts {
                    match node {
                        ASTNode::ZeroQubit { 0: ZeroQubit { sym, .. } } => {
                            result.push(*sym as char);
                        }
                        _ => {
                            return Err(PyValueError::new_err("Tensor contains non-ZeroQubit element"));
                        }
                    }
                }

                Ok(Some(result))
            }
            _ => Ok(None),
        }
    }
}
impl ASTNode {
    // Cringe helper function
    pub fn get_typ(&self) -> Option<Type> {
        match self {
            ASTNode::ZeroQubit(val) => val.typ.clone(),
            ASTNode::Tensor(val) => val.typ.clone(),
            ASTNode::OneQubit(val) => val.typ.clone(),
        }
    }

    // TODO: Fix this, not 100% sure it's right
    #[allow(unused)]
    pub fn get_dim(&self) -> Option<DimExpr> {
        match self {
            ASTNode::ZeroQubit(_) => None,
            ASTNode::Tensor(_) => None,
            ASTNode::OneQubit(val) => Some(val.factor.clone()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub elts: Vec<ASTNode>,
    pub factor: DimExpr, // For RLE (bv1 + bv2)[2], for example
    // NOTE: We want to combine all of the types of the elements; if they cannot be combined, the
    // tensor is not well typed
    pub typ: Option<Type>,
}
