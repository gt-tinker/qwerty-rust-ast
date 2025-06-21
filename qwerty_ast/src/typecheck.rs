//! Qwerty typechecker implementation: walks the AST and enforces all typing rules.

use crate::ast::*;
use crate::error::{TypeError, TypeErrorKind};
use std::collections::HashMap;

//
// ─── TYPE ENVIRONMENT ───────────────────────────────────────────────────────────
//

/// Tracks variable bindings (and potentially functions, quantum registers, etc.)
#[derive(Debug, Clone)]
pub struct TypeEnv {
    vars: HashMap<String, Type>,
    // TODO: Extend as needed (functions, modules, scopes, etc.)
}

impl TypeEnv {
    pub fn new() -> Self {
        Self { vars: HashMap::new() }
    }

    // Allows Shadowing
    // (If a variable with the same name as lhs already exists in the environment, this will overwrite (shadow) the previous binding with the new type)
    pub fn insert_var(&mut self, name: &str, typ: Type) {
        self.vars.insert(name.to_string(), typ);
    }

    // QWERTY follows Python's variable rules: shadowing is allowed.
    // To disallow shadowing, uncomment the code below and update call sites.
    
    // Disallow Shadowing (TODO: Seems not required, confirm with Austin)
    /*
    pub fn insert_var(&mut self, name: &str, typ: Type) -> Result<(), TypeError> {
        if self.vars.contains_key(name) {
            return Err(TypeError {
                kind: TypeErrorKind::RedefinedVariable(name.to_string()),
                span: None,
            });
        }
        self.vars.insert(name.to_string(), typ);
        Ok(())
    }
    // Update Usage:
    // In typecheck_function and typecheck_stmt:
    // >>    env.insert_var(name, ty.clone())?;
    */

    pub fn get_var(&self, name: &str) -> Option<&Type> {
        self.vars.get(name)
    }
}

//
// ─── TOP-LEVEL TYPECHECKER ──────────────────────────────────────────────────────
//

/// Entry point: checks the whole program.
/// Returns Ok(()) if well-typed, or a TypeError at the first mistake (Fail fast!!)
/// TODO: (Future-work!) Change it to Multi/Batch Error reporting Result<(), Vec<TypeError>>
pub fn typecheck_program(prog: &Program) -> Result<(), TypeError> {
    for func in &prog.funcs {
        typecheck_function(func)?;
    }
    Ok(())
}

/// Typechecks a single function and its body.
pub fn typecheck_function(func: &FunctionDef) -> Result<(), TypeError> {

    assert_eq!(
        func.is_rev,
        matches!(func.ret_type, Type::RevFuncType { .. }),
        "is_rev and ret_type are out of sync for function: {}",
        func.name
    );

    // ---------------------------
    let mut env = TypeEnv::new();

    // Bind function arguments in the environment.
    for (ty, name) in &func.args {
        env.insert_var(name, ty.clone());
    }

    // Track the expected return type (for return statements).
    let expected_ret_type = &func.ret_type;

    // Typecheck each statement.
    for stmt in &func.body {
        typecheck_stmt(stmt, &mut env, expected_ret_type)?;
    }

    Ok(())
}

//
// ─── STATEMENTS ────────────────────────────────────────────────────────────────
//

/// Typecheck a statement.
/// - env: The current variable/type environment.
/// - expected_ret_type: Used to check Return statements.
pub fn typecheck_stmt(
    stmt: &Stmt,
    env: &mut TypeEnv,
    expected_ret_type: &Type,
) -> Result<(), TypeError> {
    match stmt {
        Stmt::Assign { lhs, rhs, span } => {
            let rhs_ty = typecheck_expr(rhs, env)?;
            env.insert_var(lhs, rhs_ty); // Shadowing allowed for now.
            Ok(())
        }

        // TODO: Done, VERIFY!
        /*
        RHS can be of type UnitType, FuncType or RegType, but we want to only allow Unpacking for RegType,
        if it's not, thow a type Error!

            match rhs_ty {
                Type::RegType { element_type, dim } => { ... } // valid!
                _ => Err(TypeError { ... }) // all other types are errors
            }
        
        After verifying that rhs is a register of the correct dimension (e.g., qubit[3] 
        and lhs is ["a", "b", "c"]), you must assign a type to each new variable on the left.
        Each variable (a, b, c) gets its own type binding in the environment:
        */
        Stmt::UnpackAssign { lhs, rhs, span } => {
            // 1. Typecheck the right-hand side expression (should return its type)
            let rhs_ty = typecheck_expr(rhs, env)?;
            // 2. Only allow unpacking if rhs_ty is a register type (Type::RegType)
            match rhs_ty {
                Type::RegType { elem_ty, dim } => {
                    // 3. Check that number of variables matches register dimension
                    if lhs.len() as u64 != dim {
                        return Err(TypeError {
                            kind: TypeErrorKind::WrongArity { 
                                expected: dim as usize, 
                                found: lhs.len() 
                            },
                            span: span.clone(),
                        });
                    }
                    // 4. Bind each variable to a single-element register of same kind, of type RegType{ elem_ty, dim: 1 }
                    for var in lhs {
                        env.insert_var(
                            var,
                            Type::RegType {
                                elem_ty: elem_ty.clone(),
                                dim: 1,
                            },
                        );
                    }
                    Ok(())
                }
                // 5. If rhs_ty is NOT a register, raise error
                _ => Err(TypeError {
                    kind: TypeErrorKind::InvalidType(format!(
                        "Can only unpack from register type, found: {:?}", rhs_ty
                    )),
                    span: span.clone(),
                }),
            }
        }


        Stmt::Return { val, span } => {
            let val_ty = typecheck_expr(val, env)?;
            if &val_ty != expected_ret_type {
                return Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: format!("{:?}", expected_ret_type),
                        found: format!("{:?}", val_ty),
                    },
                    span: span.clone(),
                });
            }
            Ok(())
        }
    }
}

//
// ─── EXPRESSIONS ────────────────────────────────────────────────────────────────
//

/// Typecheck an expression and return its type.
pub fn typecheck_expr(expr: &Expr, env: &mut TypeEnv) -> Result<Type, TypeError> {
    match expr {
        Expr::Variable { name, span } => {
            env.get_var(name)
                .cloned()
                .ok_or(TypeError {
                    kind: TypeErrorKind::UndefinedVariable(name.clone()),
                    span: span.clone(),
                })
        }

        Expr::UnitLiteral { span: _ } => Ok(Type::UnitType),

        Expr::Adjoint { func, span } => {
            let func_ty = typecheck_expr(func, env)?;

            match func_ty {
                Type::RevFuncType { in_out_ty } => {
                    Ok(Type::RevFuncType { in_out_ty })
                }
                _ => Err(TypeError {
                    kind: TypeErrorKind::InvalidQubitOperation(
                        "Adjoint can only be applied to reversible (RevFuncType) quantum functions".to_string(),
                    ),
                    span: span.clone(),
                }),
            }
        }


        Expr::Pipe { lhs, rhs, span: _ } => {
            // Typing rule: lhs type must match rhs function input type.
            let lhs_ty = typecheck_expr(lhs, env)?;
            let rhs_ty = typecheck_expr(rhs, env)?;

            match &rhs_ty {
                Type::FuncType { in_ty, out_ty } => {
                    if **in_ty != lhs_ty {
                        return Err(TypeError {
                            kind: TypeErrorKind::MismatchedTypes {
                                expected: format!("{:?}", in_ty),
                                found: format!("{:?}", lhs_ty),
                            },
                            span: None,
                        });
                    }
                    Ok((**out_ty).clone())
                }
                
                _ => Err(TypeError {
                    kind: TypeErrorKind::NotCallable(format!("{:?}", rhs_ty)),
                    span: None,
                }),
            }
        }

        Expr::Measure { basis, span: _ } => {
            // Qwerty: measurement returns classical result; basis must be valid.
            typecheck_basis(basis, env)?; //  is it a legal quantum basis?

            Ok(Type::RegType {
                elem_ty: RegKind::Bit,
                dim: 1, // TODO: Make dynamic based on basis (Check with Austin about its validation)
                // Self Note (verify with Austin): Currently this measurement returns a single classical bit
                // But in real quantum programs, we might measure multiple qubits at once (e.g measuring a register of 3 qubits gives you 3 classical bits, ryt?)
                // The number of bits returned should depend on the size/dimension of the basis being measured
                // So, make 'dim' reflect the actual number of qubits measured, as determined by the basis argument
                // Need to understand and work on dimensions.. Discuss bro!
            })
        }

        Expr::Discard { span: _ } => Ok(Type::UnitType),

        // A tensor (often a tensor product) means combining multiple quantum states or registers into a larger, composite system
        Expr::Tensor { vals, span: _ } => {
            // => All sub-expressions in a tensor must have the same type (e.g. all are qubits, or all are bits)
            // Tensor([Qubit, Qubit, Qubit])
            let mut t = None;
            for v in vals {
                let vty = typecheck_expr(v, env)?;

                if let Some(prev) = &t {
                    if &vty != prev {
                        // if current_type doesn't match with prev_type, error!
                        return Err(TypeError {
                            kind: TypeErrorKind::MismatchedTypes {
                                expected: format!("{:?}", prev),
                                found: format!("{:?}", vty),
                            },
                            span: None,
                        });
                    }
                }
                t = Some(vty);
            }
            Ok(t.unwrap_or(Type::UnitType))
        }

        Expr::BasisTranslation { bin, bout, span: _ } => {
            // TODO: Ensure translation is between compatible bases.
            /*
            0) ASK Austin!
            1) Typecheck both bases (already done)
            2) Extract relevant info from each basis (e.g. dimension, type).
            3) Compare the properties we care about (e.g. same dimension, same qubit type).
            4) Return an error if they are not compatible.
            */
            typecheck_basis(bin, env)?;
            typecheck_basis(bout, env)?;
            Ok(Type::UnitType)
        }

        Expr::Predicated {
            then_func,
            else_func,
            pred,
            span: _,
        } => {
            let t_ty = typecheck_expr(then_func, env)?;
            let e_ty = typecheck_expr(else_func, env)?;
            let _pred_ty = typecheck_basis(pred, env)?;
            if t_ty != e_ty {
                return Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: format!("{:?}", t_ty),
                        found: format!("{:?}", e_ty),
                    },
                    span: None,
                });
            }
            Ok(t_ty)
        }

        Expr::NonUniformSuperpos { pairs, span: _ } => {
            // Each pair is (weight, QLit). All QLits must have same type.
            let mut qt = None;
            for (_, qlit) in pairs {
                let qlit_ty = typecheck_qlit(qlit, env)?;
                if let Some(prev) = &qt {
                    if &qlit_ty != prev {
                        return Err(TypeError {
                            kind: TypeErrorKind::MismatchedTypes {
                                expected: format!("{:?}", prev),
                                found: format!("{:?}", qlit_ty),
                            },
                            span: None,
                        });
                    }
                }
                qt = Some(qlit_ty);
            }
            Ok(qt.unwrap_or(Type::UnitType))
        }

        Expr::Conditional {
            then_expr,
            else_expr,
            cond,
            span: _,
        } => {
            let t_ty = typecheck_expr(then_expr, env)?;
            let e_ty = typecheck_expr(else_expr, env)?;
            let _c_ty = typecheck_expr(cond, env)?;
            if t_ty != e_ty {
                return Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: format!("{:?}", t_ty),
                        found: format!("{:?}", e_ty),
                    },
                    span: None,
                });
            }
            Ok(t_ty)
        }

        Expr::QLit(qlit) => typecheck_qlit(qlit, env),
    }
}

//
// ─── QLIT, VECTOR, BASIS HELPERS ──────────────────────────────────────────────
//

/// Typecheck a QLit node.
/// TODO: Enforce Qwerty rules about QLit types and quantum registers.
fn typecheck_qlit(qlit: &QLit, _env: &mut TypeEnv) -> Result<Type, TypeError> {
    match qlit {
        QLit::ZeroQubit { .. } | QLit::OneQubit { .. } => Ok(Type::RegType {
            elem_ty: RegKind::Qubit,
            dim: 1,
        }),

        QLit::QubitTilt { q, .. } => typecheck_qlit(q, _env),

        QLit::UniformSuperpos { q1, q2, .. } => {
            let t1 = typecheck_qlit(q1, _env)?;
            let t2 = typecheck_qlit(q2, _env)?;
            if t1 == t2 {
                Ok(t1)
            } else {
                Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: format!("{:?}", t1),
                        found: format!("{:?}", t2),
                    },
                    span: None,
                })
            }
        }

        QLit::QubitTensor { qs, .. } => {
            // TODO: Combine types; for now, just check all are Qubits.
            for q in qs {
                let t = typecheck_qlit(q, _env)?;
                if t != (Type::RegType { elem_ty: RegKind::Qubit, dim: 1 }) {
                    return Err(TypeError {
                        kind: TypeErrorKind::InvalidQubitOperation(format!("{:?}", t)),
                        span: None,
                    });
                }
            }
            Ok(Type::RegType {
                elem_ty: RegKind::Qubit,
                // dim: qs.len() as u32,
                dim: qs.len() as u64,
            })
        }
    }
}

/// Typecheck a Vector node (see grammar for rules).
fn typecheck_vector(vector: &Vector, _env: &mut TypeEnv) -> Result<Type, TypeError> {
    match vector {
        Vector::ZeroVector { .. }
        | Vector::OneVector { .. }
        | Vector::PadVector { .. }
        | Vector::TargetVector { .. } => Ok(Type::UnitType), // TODO: clarify

        Vector::VectorTilt { q, .. } => typecheck_qlit(q, _env),

        Vector::UniformVectorSuperpos { q1, q2, .. } => {
            let t1 = typecheck_qlit(q1, _env)?;
            let t2 = typecheck_qlit(q2, _env)?;
            if t1 == t2 {
                Ok(t1)
            } else {
                Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: format!("{:?}", t1),
                        found: format!("{:?}", t2),
                    },
                    span: None,
                })
            }
        }

        Vector::VectorTensor { qs, .. } => {
            for q in qs {
                let t = typecheck_qlit(q, _env)?;
                if t != (Type::RegType { elem_ty: RegKind::Qubit, dim: 1 }) {
                    return Err(TypeError {
                        kind: TypeErrorKind::InvalidQubitOperation(format!("{:?}", t)),
                        span: None,
                    });
                }
            }
            Ok(Type::RegType {
                elem_ty: RegKind::Qubit,
                dim: qs.len() as u64,
            })
        }
    }
}

/// Typecheck a Basis node.
/// TODO: Enforce more quantum rules as per Qwerty basis specification.
fn typecheck_basis(basis: &Basis, env: &mut TypeEnv) -> Result<Type, TypeError> {
    match basis {
        Basis::BasisLiteral { vecs, .. } => {
            for v in vecs {
                typecheck_vector(v, env)?;
            }
            Ok(Type::UnitType) // TODO: Should this return a Basis type?
        }

        Basis::EmptyBasisLiteral { .. } => Ok(Type::UnitType),

        Basis::BasisTensor { bases, .. } => {
            for b in bases {
                typecheck_basis(b, env)?;
            }
            Ok(Type::UnitType)
        }
    }
}

//
// ─── UNIT TESTS ────────────────────────────────────────────────────────────────
//

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    #[test]
    fn test_typecheck_var_and_assign() {
        let prog = Program {
            funcs: vec![
                FunctionDef::new(
                    "main".into(),
                    vec![(Type::UnitType, "x".into())],
                    Type::UnitType,
                    vec![
                        Stmt::Assign {
                            lhs: "y".into(),
                            rhs: Expr::Variable {
                                name: "x".into(),
                                span: None,
                            },
                            span: None,
                        }
                    ],
                    None,
                )
            ],
            span: None,
        };

        let result = typecheck_program(&prog);
        assert!(result.is_ok());
    }

    #[test]
    fn test_unpack_assign_typing() {
        use crate::ast::*;

        // Case 1: Legal unpack (2 variables, qubit[2])
        let legal_prog = Program {
            funcs: vec![FunctionDef::new (
                "main".into(),
                vec![],
                Type::UnitType,
                vec![
                    Stmt::UnpackAssign {
                        lhs: vec!["a".into(), "b".into()],
                        rhs: Expr::Variable {
                            name: "payload".into(),
                            span: None,
                        },
                        span: None,
                    }
                ],
                None,
                )],
            span: None,
        };

        // The environment must bind payload to a qubit[2] for this to pass
        let mut env = TypeEnv::new();
        env.insert_var("payload", Type::RegType { elem_ty: RegKind::Qubit, dim: 2 });

        // The typechecker expects to build the environment itself, so let's test the stmt directly:
        let stmt = &legal_prog.funcs[0].body[0];
        let result = super::typecheck_stmt(stmt, &mut env, &Type::UnitType);
        assert!(result.is_ok(), "Legal unpack failed typechecking");

        // Case 2: Illegal unpack (3 variables, qubit[2])
        let illegal_stmt = Stmt::UnpackAssign {
            lhs: vec!["a".into(), "b".into(), "c".into()],
            rhs: Expr::Variable {
                name: "payload".into(),
                span: None,
            },
            span: None,
        };
        let mut env2 = TypeEnv::new();
        env2.insert_var("payload", Type::RegType { elem_ty: RegKind::Qubit, dim: 2 });

        let result2 = super::typecheck_stmt(&illegal_stmt, &mut env2, &Type::UnitType);
        assert!(
            matches!(result2, Err(TypeError { kind: TypeErrorKind::WrongArity { .. }, .. })),
            "Unpack of mismatched arity did not fail as expected"
        );
    }

    #[test]
    fn test_unpack_assign_non_register_rhs() {
        use crate::ast::*;

        // Trying to unpack a UnitType (not a register!)
        let stmt = Stmt::UnpackAssign {
            lhs: vec!["a".into()],
            rhs: Expr::Variable {
                name: "not_a_reg".into(),
                span: None,
            },
            span: None,
        };
        let mut env = TypeEnv::new();
        env.insert_var("not_a_reg", Type::UnitType);

        let result = super::typecheck_stmt(&stmt, &mut env, &Type::UnitType);
        assert!(
            matches!(result, Err(TypeError { kind: TypeErrorKind::InvalidType(_), .. })),
            "Unpack of non-register type did not fail as expected"
        );
    }

    // TODO: Add more tests for all language constructs! In separate test file? Later
}
