#![allow(unused_imports, dead_code)]
use crate::dimexpr::*;
/// The goal of this file is to define all of the structs, enums, types, etc., related to Bases and
/// Basis literals
// Also pay special attention to singleton basis stuff

// How do we want to represent a single eigenvector/basis literal, then a singleton basis (with run
// length encoding?), and combinations of basis literals such as
// '0' + 'p'
use crate::types::{Eigenbit, PrimitiveBasis, Type};
use std::any::Any;


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
    pub typ: Option<Type>
}

// Something like
// STD + Zero = 0
// PM  + Zero = +
// IJ  + One  = j
// TODO: Typecheck for Fourier as the exception
#[derive(Clone, Debug)]
pub struct QubitSymbol {
    pub sym: u8,
    pub typ: Option<Type>
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
