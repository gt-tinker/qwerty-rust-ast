#![allow(unused_imports, dead_code)]
use crate::ast::DebugInfo;
use crate::dimexpr::*;
/// The goal of this file is to define all of the structs, enums, types, etc., related to Bases and
/// Basis literals
// Also pay special attention to singleton basis stuff

// How do we want to represent a single eigenvector/basis literal, then a singleton basis (with run
// length encoding?), and combinations of basis literals such as
// '0' + 'p'
use crate::types::{Eigenbit, PrimitiveBasis, Type};
use std::any::Any;

// This defines the basis type (separate type, so important it deserves its own file!)
// It should be a vector of basis literal elements

// {bv1, bv2} + {bv3, bv4} + ...
#[derive(Clone, Debug)]
pub struct Basis {
    pub elts: Vec<BasisElement>,
    pub dim: DimExpr, // Represents the sum of all of the elements' dims?
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}

// FIXME: qwerty.QwertyTypeError: Qubit literals in a basis literal must not mix vectors of std[N], ij[N], and pm[N], but a vector from pm[N] and a vector from std[N] are mixed in the vector at index 0 (at column 12)
// We need to correctly handle this error, which we currently do not. THIS IS A NOTE TO SELF

// BasisElement is a trait (idea)

#[derive(Clone, Debug)]
pub enum BasisElement {
    BuiltInBasis(BuiltInBasis),
    BasisLiteral(BasisLiteral),
}

// NOTE: BasisVector is usually {bv1, bv2, ...}

// I basically copied this from the preprint lang spec
#[derive(Clone, Debug)]
pub struct BasisLiteral {
    pub literal: QubitLiteral,
    pub phase: Option<f64>, // TODO: Fix this, can be an expression
    // NOTE: We will have to modify the frontend to make the
    // phase an f64
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}

// TODO: Add more stuff from the preprint lang spec

// This is for constructs like std[N]
#[derive(Clone, Debug)]
pub struct BuiltInBasis {
    pub prim_basis: PrimitiveBasis,
    pub dim: DimExpr,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}

// NOTE: Differs from current implementation, supports
// {'+0', '-1'}
//
// Elementwise primitive bases need to match
// aka {'+0'} >> {'11'} is not allowed
// BANNED
// NOTE: Need to enable span checking and
// orthonormality checking!!

/// To represent something like '01pm'[5]
#[derive(Clone, Debug)]
pub struct QubitLiteral {
    pub elts: Vec<QubitSymbol>,
    pub factor: DimExpr,
    pub typ: Option<Type>,
    pub dbg: DebugInfo,
}

// Something like
// STD + Zero = 0
// PM  + Zero = +
// IJ  + One  = j
// TODO: Typecheck for Fourier as the exception
#[derive(Clone, Debug)]
pub struct QubitSymbol {
    pub basis: PrimitiveBasis,
    pub eigenbit: Eigenbit,
    pub dbg: DebugInfo,
}

// NOTE: Removed type from QubitSymbol since all QubitSymbols will have
// type Qubit[1]

// Examples of basis elements:
// 1. '0', '1', 'p', 'm', etc.
// 2. '10', 'p0', 'm101', etc.
// 3. { '00', '11'}, etc.
// NOTE: '0' is desugaring of { '0' }
// 4. 'p'[N] (dimvars) (singleton basis)
// 5. Do we count std, pm, ij, fourier, etc?
