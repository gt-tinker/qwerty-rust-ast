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


#[pymodule]
fn qwerty_ast(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NodeBox>()?;
    Ok(())
}