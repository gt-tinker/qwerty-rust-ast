use crate::span::SourceSpan;

/// UndefinedVariable: Variable is used but not declared or is out of scope.
/// RedefinedVariable: Variable declared more than once in the same scope.
/// UninitializedVariable: Variable is used before being assigned a value.
/// ImmutableAssignment: Attempt to assign to a variable that is immutable.
/// MismatchedTypes: Value type does not match the expected type.
/// WrongArity: Number of arguments in a call does not match what is expected.
/// NotCallable: Attempt to call something that is not a function or gate.
/// InvalidType: Type is not supported or is malformed.
/// InvalidOperation: Operation is not valid for the given type.
/// TypeInferenceFailure: Compiler could not determine the type.
/// QuantumMeasurementOnClassical: Attempt to measure a classical (non-qubit) variable.
/// InvalidQubitOperation: Invalid operation performed on a qubit.
/// UnsupportedPythonConstruct: Python AST node not supported by the DSL.
/// NonReversibleOperationInReversibleFunction: Function is declared @reversible but contains operations that are inherently non-reversible.
/// ReversibilityAnnotationMismatch: Function's declared @reversible annotation and its return type (RevFuncType) are inconsistent.

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
    // Quantum-specific errors:
    QuantumMeasurementOnClassical(String),
    InvalidQubitOperation(String),
    // Python-DSL/AST-specific errors:
    UnsupportedPythonConstruct(String),
    SyntaxError(String),

    NonReversibleOperationInReversibleFunction(String),
    ReversibilityAnnotationMismatch {
        declared_reversible: bool, // From annotation (func.is_rev)
        inferred_reversible: bool, // From signature (matches!(func.ret_type, Type::RevFuncType))
        func_name: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub span: Option<SourceSpan>,
}
