#![allow(unused_imports, dead_code)]

/// This module serves as a testbed and design area for the new Qwerty AST, built in Rust
/// The goal of this particular file is to test implementations that make the AST the
/// most ergonomic to use!

use pyo3::prelude::*;
use ast::ASTNode;
use dimexpr::DimExpr;
use dimexpr::*;
use inference::{ConstraintInfo, Context, DimConstraint};
use std::collections::HashMap;
use std::str::FromStr;
use types::{
    Eigenbit, FuncType, Order, PrimitiveBasis, RegisterType, RegisterVariant, Type, UnitType,
};
use crate::ast::NodeBox;

mod ast;
mod basis;
mod dimexpr;
mod inference;
mod types;



macro_rules! ast_node {
        ($id:ident, $( $field:ident : $val:expr ),*) => {
            Box::new(ASTNode::$id(ast::$id {
           pub(crate)     $( $field : $val, )*
            }))
        };
    }

macro_rules! ast_node_struct {
        ($id:ident, $( $field:ident : $val:expr ),*) => {
            ast::$id {
                $( $field : $val, )*
            };
        };
    }

macro_rules! ast_dim {
        ($dim_str:expr) => {
            dimexpr::DimExpr::from_str($dim_str).unwrap()
        };
    }

#[pymodule]
fn qwerty_ast(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NodeBox>()?;
    Ok(())
}