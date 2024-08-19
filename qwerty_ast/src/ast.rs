//! This is a proof-of-concept for a simple enum and struct based AST
//! structure, with natural nested nodes and fat pointers/Boxes for
//! holding child nodes

// FIXME: Remove later!
#![allow(unused_imports, dead_code)]
use crate::basis::*;
use crate::dimexpr::{DimExpr, DimVar};
use crate::inference::{ConstraintInfo, Context, DimConstraint};
use crate::types::Type;
use crate::types::*;
use num::complex::Complex;
use pyo3::prelude::*;
use pyo3::types::PyFrame;
use std::collections::{HashMap, HashSet};

pub use crate::basis::{Basis, BuiltInBasis, QubitLiteral, QubitSymbol};

#[derive(Debug, Clone)]
pub enum ASTNode {
    Dagger(Dagger),
    Prepare(Prepare),
    LiftBits(LiftBits),
    EmbedClassical(EmbedClassical),
    Pipe(Pipe),
    Grab(Grab),
    QubitLiteral(QubitLiteral),
    Phase(Phase),
    FloatLiteral(FloatLiteral),
    FloatNeg(FloatNeg),
    FloatBinaryOp(FloatBinaryOp),
    FloatDimExpr(FloatDimExpr),
    // TupleLiteral(TupleLiteral),
    Tensor(Tensor),
    BuiltInBasis(BuiltInBasis),
    Basis(Basis), // TODO: Necessary?
    Identity(Identity),
    BasisTranslation(BasisTranslation),
    Discard(Discard),
    Measure(Measure),
    Project(Project),
    Flip(Flip),
    Rotate(Rotate),
    BasisLiteral(BasisLiteral),
    Conditional(Conditional),
    Return(Return),
    Pred(Pred),
    Program(Program),
    Function(Function),
    Call(Call),
    Assign(Assign),
    Variable(String),
}

impl ASTNode {
    // Cringe helper function
    pub fn get_typ(&self) -> Option<Type> {
        match self {
            ASTNode::Dagger(val) => val.typ.clone(),
            ASTNode::Prepare(val) => val.typ.clone(),
            ASTNode::LiftBits(val) => val.typ.clone(),
            ASTNode::EmbedClassical(val) => val.typ.clone(),
            ASTNode::Pipe(val) => val.typ.clone(),
            ASTNode::Grab(val) => val.typ.clone(),
            ASTNode::QubitLiteral(val) => val.typ.clone(),
            ASTNode::Phase(_) => None,
            ASTNode::FloatLiteral(val) => val.typ.clone(),
            ASTNode::FloatNeg(val) => val.typ.clone(),
            ASTNode::FloatBinaryOp(val) => val.typ.clone(),
            ASTNode::FloatDimExpr(val) => val.typ.clone(),
            // ASTNode::TupleLiteral(val) => val.typ.clone(),
            ASTNode::Tensor(val) => val.typ.clone(),
            ASTNode::BuiltInBasis(val) => val.typ.clone(),
            ASTNode::Basis(val) => val.typ.clone(), // TODO: Necessary?
            ASTNode::Identity(val) => val.typ.clone(),
            ASTNode::BasisTranslation(val) => val.typ.clone(),
            ASTNode::Discard(val) => val.typ.clone(),
            ASTNode::Measure(val) => val.typ.clone(),
            ASTNode::Project(val) => val.typ.clone(),
            ASTNode::Flip(val) => val.typ.clone(),
            ASTNode::Rotate(val) => val.typ.clone(),
            ASTNode::BasisLiteral(val) => val.typ.clone(),
            ASTNode::Conditional(val) => val.typ.clone(),
            ASTNode::Return(val) => val.typ.clone(),
            ASTNode::Pred(val) => val.typ.clone(),
            ASTNode::Program(_) => None,
            ASTNode::Function(val) => val.typ.clone(),
            ASTNode::Call(val) => val.typ.clone(),
            ASTNode::Assign(_) => None,
            ASTNode::Variable(_) => None,
        }
    }

    // TODO: Fix this, not 100% sure it's right
    #[allow(unused)]
    pub fn get_dim(&self) -> Option<DimExpr> {
        match self {
            ASTNode::Dagger(val) => None,
            ASTNode::Prepare(val) => None,
            ASTNode::LiftBits(val) => None,
            ASTNode::EmbedClassical(val) => None,
            ASTNode::Pipe(val) => Some(val.lhs.get_dim()?),
            ASTNode::Grab(val) => None,
            ASTNode::QubitLiteral(val) => Some(val.factor.clone()), // TODO: Should be multiplied
            // by the number of elts (this is true for a few of the other nodes)
            ASTNode::Phase(_) => None,
            ASTNode::FloatLiteral(val) => None,
            ASTNode::FloatNeg(val) => None,
            ASTNode::FloatBinaryOp(val) => None,
            ASTNode::FloatDimExpr(val) => Some(val.value.clone()),
            // ASTNode::TupleLiteral(val) => val.typ.clone(),
            ASTNode::Tensor(val) => Some(val.factor.clone()),
            ASTNode::BuiltInBasis(val) => Some(val.dim.clone()),
            ASTNode::Basis(val) => Some(val.dim.clone()), // TODO: Necessary?
            ASTNode::Identity(val) => None,
            ASTNode::BasisTranslation(val) => Some(val.basis_in.get_dim()?),
            ASTNode::Discard(val) => None, // TODO: unsure
            ASTNode::Measure(val) => Some(val.basis.get_dim()?),
            ASTNode::Project(val) => Some(val.basis.get_dim()?),
            ASTNode::Flip(val) => Some(val.basis.get_dim()?),
            ASTNode::Rotate(val) => Some(val.basis.get_dim()?),
            ASTNode::BasisLiteral(val) => Some(DimExpr::from(1)),
            ASTNode::Conditional(val) => Some(val.then_expr.get_dim()?),
            ASTNode::Return(val) => Some(val.value.get_dim()?),
            ASTNode::Pred(val) => Some(val.basis.get_dim()?),
            ASTNode::Program(_) => None,
            ASTNode::Function(val) => None,
            ASTNode::Call(val) => None,
            ASTNode::Assign(_) => None,
            ASTNode::Variable(_) => None,
        }
    }
}

// NOTE: Structs for representing program structure!
// (a.k.a. the Python-isms, in some sense)

#[derive(Clone, Debug)]
pub struct Program {
    pub body: Vec<ASTNode>,
    // Context is the mapping that stores all of the relevant types
    pub ctx: Context,
    pub dbg: DebugInfo,
}

// NOTE: Need to have information on input and return types
#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub args: Vec<ASTNode>,
    pub body: Vec<ASTNode>,
    pub is_classical: bool, // Also contained within the type, do we need?
    pub constraint_info: ConstraintInfo,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}

#[derive(Clone, Debug)]
pub struct Call {
    pub name: String,
    // NOTE: We type check the ins against
    // the type found in the env, for the corresponding
    // function name
    pub ins: Vec<ASTNode>, // Can be a value or a variable
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}

#[derive(Clone, Debug)]
pub struct Assign {
    pub assigned_to: Vec<String>, // list of variables to assign to
    pub val: Box<ASTNode>,        // should this be a vector?
    pub dbg: DebugInfo,
}

// Additional NOTE: Using macros to streamline repeated Trait
// implementations is fun and cool

// Struct definitions (from defs.hpp, ast.hpp)
// TODO: Add examples of each struct's syntactic structure!
// As an example, see `Dagger` node

// NOTE: We use Option<Box<dyn Type>> for dynamic dispatch (vtable magic)

/// The `~` adjoint operation
#[derive(Debug, Clone)]
pub struct Dagger {
    pub operand: Box<ASTNode>,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct Prepare {
    pub operand: Box<ASTNode>,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct LiftBits {
    pub bits: Box<ASTNode>,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct EmbedClassical {
    pub name: String,
    pub operand_name: String,
    pub embed_kind: EmbeddingKind,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct Pipe {
    pub lhs: Box<ASTNode>,
    pub rhs: Box<ASTNode>,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct Grab {
    pub var: Box<ASTNode>,
    pub grab_vals: Vec<DimExpr>,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}

#[derive(Debug, Clone)]
pub struct Repeat {
    pub body: Box<ASTNode>,
    pub loopvar: String,
    pub ub: DimExpr, // note to self: upper bound
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}
// NOTE: Phasing out for QubitLiteral
// pub struct BiTensor {
//     pub left: Box<ASTNode>,
//     pub right: Box<ASTNode>,
//     pub typ: Option<Box<dyn Type>>,
//     pub only_literals: bool,
//     pub singleton_basis: bool,
//     pub dbg: DebugInfo,
// }

#[derive(Debug, Clone)]
pub struct Tensor {
    pub elts: Vec<ASTNode>,
    pub factor: DimExpr, // For RLE (bv1 + bv2)[2], for example
    // NOTE: We want to combine all of the types of the elements; if they cannot be combined, the
    // tensor is not well typed
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}

// NOTE: Where is this used, usually? Do we need this?
// pub struct BroadcastTensor {
//     pub value: Box<ASTNode>,
//     pub factor: DimExpr,
//     pub typ: Option<Box<dyn Type>>,
//     pub only_literals: bool,
//     pub singleton_basis: bool,
//     pub dbg: DebugInfo,
// }
// NOTE: We now use the QubitLiteral struct from the
// basis file/module
// pub struct QubitLiteral {
//     eigenstate: Eigenstate,
//     pauli: Pauli,
//     dim: usize,
//     dbg: DebugInfo,
// }
#[derive(Debug, Clone)]
pub struct Phase {
    pub phase: Box<ASTNode>,
    pub value: Box<ASTNode>,
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct FloatLiteral {
    pub value: f64,
    pub typ: Option<Type>, // Type::Float
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct FloatNeg {
    pub operand: Box<ASTNode>,
    pub typ: Option<Type>, // Type::Float
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct FloatBinaryOp {
    pub op: FloatOp,
    pub left: Box<ASTNode>,
    pub right: Box<ASTNode>,
    pub typ: Option<Type>, // Type::Float
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct FloatDimExpr {
    pub value: DimExpr,
    pub typ: Option<Type>, // Type::Float
    pub dbg: DebugInfo,
}

// list of concatenated ASTNodes
#[derive(Debug, Clone)]
pub struct TupleLiteral {
    pub elts: Vec<Box<ASTNode>>,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}
// NOTE: Use BuiltInBasis from basis file
// pub struct BuiltInBasis {
//     pauli: Pauli,
//     basis: BuiltInBasis, // NOTE: Each variant also can store the dimension, like Z(1) or
//     // Fourier(5)
//     dbg: DebugInfo,
// }
#[derive(Debug, Clone)]
pub struct Identity {
    pub typ: Option<Type>, // NOTE: Type::FuncType
    pub dim: DimExpr,      // for id[N+2], for example
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct BasisTranslation {
    pub basis_in: Box<ASTNode>,
    pub basis_out: Box<ASTNode>,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct Discard {
    pub typ: Option<Type>, // NOTE: Type::FuncType
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct Measure {
    pub basis: Box<ASTNode>,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct Project {
    pub basis: Box<ASTNode>,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct Flip {
    pub basis: Box<ASTNode>,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct Rotate {
    pub basis: Box<ASTNode>,
    pub theta: Box<ASTNode>,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}
// NOTE: Use from basis.rs
// pub struct BasisLiteral {
//     elts: Vec<Box<ASTNode>>,
//     typ: Option<Box<dyn Type>>,
//     dbg: DebugInfo,
// }
#[derive(Debug, Clone)]
pub struct Conditional {
    pub if_expr: Box<ASTNode>,
    pub then_expr: Box<ASTNode>,
    pub else_expr: Box<ASTNode>,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct Return {
    pub value: Box<ASTNode>,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}
#[derive(Debug, Clone)]
pub struct Pred {
    pub order: Order,
    pub basis: Box<ASTNode>,
    pub body: Box<ASTNode>,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}

// TODO: See if we can use the proc_macro's Span struct later
// TODO: Create a constructor for this!
#[derive(Default, Debug, Clone)]
pub struct DebugInfo {
    file: String, // file!() macro
    line: u32,    // line!() macro
    col: u32,     // column!() macro
    frame: Option<Py<PyFrame>>,
    // TODO: Add more fields if needed
}

// NOTE: Implement Eq overloading for debuginfo
