use std::fmt;
use crate::dbg::DebugLoc;

#[derive(Debug, Clone, PartialEq)]
pub enum TypeErrorKind {
    UndefinedVariable(String),
    RedefinedVariable(String),
    UninitializedVariable(String),
    ImmutableAssignment(String),
    MismatchedTypes { expected: String, found: String },
    WrongArity { expected: usize, found: usize },
    NotCallable(String),
    InvalidType(String),
    InvalidOperation { op: String, ty: String },
    TypeInferenceFailure,
    EmptyLiteral,
    DimMismatch,
    // Quantum-specific errors:
    InvalidBasis,
    NotOrthogonal { left: String, right: String },
    InvalidQubitOperation(String),
}

impl fmt::Display for TypeErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeErrorKind::UndefinedVariable(var) => write!(f, "Variable '{var}' is used but not declared or is out of scope."),
            TypeErrorKind::RedefinedVariable(var) => write!(f, "Variable '{var}' declared more than once in the same scope."),
            TypeErrorKind::UninitializedVariable(var) => write!(f, "Variable '{var}' is used before being assigned a value."),
            TypeErrorKind::ImmutableAssignment(var) => write!(f, "Attempt to assign to a variable '{var}' that is immutable."),
            TypeErrorKind::MismatchedTypes { expected, found } => write!(f, "Value type '{found}' does not match the expected type '{expected}'."),
            TypeErrorKind::WrongArity { expected, found } => write!(f, "Number of arguments in a call {found} does not match the expected number {expected}."),
            TypeErrorKind::NotCallable(what) => write!(f, "Attempt to call '{what}' that is not a function."),
            TypeErrorKind::InvalidType(ty) => write!(f, "Type '{ty}' is not supported or is malformed."),
            TypeErrorKind::InvalidOperation { op, ty } => write!(f, "Operation '{op}' is not valid for the given type '{ty}'."),
            // TODO: needs more details
            TypeErrorKind::TypeInferenceFailure => write!(f, "Compiler could not determine the type."),
            TypeErrorKind::EmptyLiteral => write!(f, "Literal is unexpectedly empty."),
            // TODO: needs more details
            TypeErrorKind::DimMismatch => write!(f, "Dimension mismatch."),
            // TODO: needs more details (maybe?)
            TypeErrorKind::InvalidBasis => write!(f, "Invalid basis."),
            // TODO: say something more specific than "constructs"?
            TypeErrorKind::NotOrthogonal { left, right } => write!(f, "The constructs '{left}' and '{right}' are not orthogonal."),
            TypeErrorKind::InvalidQubitOperation(op) => write!(f, "Invalid operation '{op}' performed on a qubit."),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub dbg: Option<DebugLoc>,
}
