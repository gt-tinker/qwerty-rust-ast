// FIXME: Remove later!
#![allow(unused_imports, dead_code)]

use crate::dimexpr::*;
use std::any::Any;
use std::collections::HashMap;

use crate::basis::{QubitLiteral, QubitSymbol};

// Copying this from ast.hpp => QubitLiteral's definition
// This selects the eigenvector based on a given PrimitiveBasis
// PrimitiveBasis PM with Eigenbit Zero is the + state
#[derive(Clone, Debug)]
pub enum Eigenbit {
    Zero,
    One,
}

// FIXME: Fix if needed!
#[derive(Clone, Debug)]
pub enum PrimitiveBasis {
    Std,
    Pm,
    Ij,
    Fourier,
}

#[derive(Clone, Debug)]
pub enum EmbeddingKind {
    EmbedBennet,
    EmbedPhase,
    EmbedInplace,
}

// NOTE: Idea, what if instead of Type being an enum, it is a Trait, and we have trait
// implementations for structs like float, functype, etc.

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    UnitType(),
    RegisterType(RegisterType),
    FuncType(FuncType),
    FloatType(FloatType),
    IntType(IntType),
    ComplexType(ComplexType),
    BasisType(BasisType),
    TypeVar(TypeVar),
}

impl Type {
    fn is_subtype_of(&self, other: &Type) -> bool {
        match self {
            Type::UnitType() => {
                if let Type::RegisterType(_) = other {
                    true
                } else {
                    false
                }
            }
            Type::RegisterType(val) => {
                if let Type::RegisterType(val2) = other {
                    val.variant == val2.variant && val.dim == val2.dim
                } else {
                    false
                }
            }
            Type::FuncType(val) => {
                // FIXME: !!!
                if let Type::FuncType(val2) = other {
                    val.lhs.is_subtype_of(&val2.lhs)
                        && val.rhs.is_subtype_of(&val2.rhs)
                        && (!val2.is_rev || val.is_rev)
                } else {
                    false
                }
            }
            Type::BasisType(val) => val.dim.is_constant(),
            Type::FloatType(_) | Type::IntType(_) | Type::ComplexType(_) => true,
            Type::TypeVar(_) => false, // Fix later!
        }
    }

    fn is_constant(&self) -> bool {
        match self {
            Type::RegisterType(val) => val.dim.is_constant(),
            Type::FuncType(val) => val.lhs.is_constant() && val.rhs.is_constant(),
            Type::BasisType(val) => val.dim.is_constant(),
            Type::FloatType(_) | Type::IntType(_) | Type::ComplexType(_) | Type::UnitType() => true,
            Type::TypeVar(_) => false,
        }
    }
    fn is_classical(&self) -> bool {
        match self {
            Type::RegisterType(val) => val.variant.is_classical(),
            Type::BasisType(_) | Type::TypeVar(_) => false,
            Type::FloatType(_)
            | Type::IntType(_)
            | Type::ComplexType(_)
            | Type::UnitType()
            | Type::FuncType(_) => true,
        }
    }
    fn is_reversible_friendly(&self) -> bool {
        match self {
            Type::RegisterType(val) => val.variant.is_reversible_friendly(),
            Type::BasisType(_) | Type::TypeVar(_) => false,
            Type::FloatType(_) | Type::IntType(_) | Type::ComplexType(_) | Type::UnitType() => true,
            Type::FuncType(val) => val.is_rev,
        }
    }
    fn is_linear(&self) -> bool {
        !self.is_classical()
    }

    pub fn get_dim(&self) -> Option<DimExpr> {
        match self {
            Type::RegisterType(val) => Some(val.dim.clone()),
            Type::BasisType(val) => Some(val.dim.clone()),
            Type::FloatType(_)
            | Type::IntType(_)
            | Type::ComplexType(_)
            | Type::UnitType()
            | Type::TypeVar(_)
            | Type::FuncType(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnitType;

// Type Variable stuff, for inference
#[derive(Clone, Debug, PartialEq)]
pub struct TypeVarGenerator {
    pub counter: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypeVar {
    pub uid: usize,
}

impl TypeVar {
    pub fn new_with(uid: usize) -> Self {
        Self { uid }
    }
}

impl TypeVarGenerator {
    pub const fn new() -> Self {
        Self { counter: 1 }
    }

    pub fn new_tyvar(&mut self) -> TypeVar {
        let var = TypeVar { uid: self.counter };
        self.counter += 1;
        var
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum RegisterVariant {
    Qubit,
    Bit,
    Angle,
}

impl RegisterVariant {
    fn is_classical(&self) -> bool {
        match &self {
            Self::Qubit => false,
            _ => true, // NOTE: Both other variants are classical indeed
        }
    }
    fn is_reversible_friendly(&self) -> bool {
        match &self {
            Self::Qubit => false,
            _ => true, // NOTE: Both other variants are reversible friendly indeed
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegisterType {
    pub variant: RegisterVariant,
    pub dim: DimExpr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FuncType {
    pub lhs: Box<Type>,
    pub rhs: Box<Type>,
    pub is_rev: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BasisType {
    pub dim: DimExpr,
}

// For Select and Pred node
// NOTE: What do these mean?
//
// See PredOrder in ast.hpp. I added a description for the artifact. Maybe Pred
// should be split into 2 different nodes. Let me know what you think. -austin
#[derive(Clone, Debug)]
pub enum Order {
    BU,
    UB,
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FloatOp {
    Div,
    Pow,
    Mul,
}

// TODO: FloatType
#[derive(Clone, Debug, PartialEq)]
pub struct FloatType {
    op: FloatOp,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IntType;

#[derive(Clone, Debug, PartialEq)]
pub struct ComplexType;

// TODO: Impl ComplexType helper functions (?)
