//! This module holds the state of evaluation in the Qwerty REPL and how steps
//! of evaluation are taken. The latter is based loosely on Appendix A of
//! arXiv:2404.12603.

use crate::ast::{Expr, QLit, Stmt};

use quantum_sparse_sim::QuantumSim;
use std::collections::HashMap;

/// Holds the quantum simulator state and a mapping of names to values.
pub struct ReplState {
    sim: QuantumSim,
    bindings: HashMap<String, Expr>, // TODO: figure out what expr should be
}

impl ReplState {
    /// Creates a new ReplState with no qubits allocated and no names bound.
    pub fn new() -> Self {
        ReplState {
            sim: QuantumSim::new(None),
            bindings: HashMap::new(),
        }
    }

    /// Evaluates an expression and returns a value.
    pub fn run(&mut self, stmt: &Stmt) -> Expr {
        // TODO: should this return a Value (if that exists) instead of an Expr? depends on the choice above

        // TODO: run this expression

        // TODO: return the value resulting from evaluation instead of copying the input
        if let Stmt::Expr { expr, .. } = stmt {
            expr.eval_to_value(self)
        } else {
            Expr::UnitLiteral { dbg: None }
        }
    }
}

impl Expr {
    pub fn is_value(&self) -> bool {
        match self {
            Expr::Variable { .. } => false,
            Expr::UnitLiteral { .. } => true,
            Expr::Adjoint { func, .. } => func.is_value(),
            Expr::Pipe { .. } => false,
            Expr::Measure { .. } => true,
            Expr::Discard { .. } => true,
            Expr::Tensor { vals, .. } => vals
                .iter()
                .all(|v| v.is_value() && !matches!(v, Expr::UnitLiteral { .. })),
            Expr::BasisTranslation { .. } => true,
            Expr::Predicated {
                then_func,
                else_func,
                ..
            } => then_func.is_value() && else_func.is_value(),
            Expr::NonUniformSuperpos { .. } => false,
            Expr::Conditional {
                then_expr,
                else_expr,
                cond,
                ..
            } => then_expr.is_value() && else_expr.is_value() && cond.is_value(),
            Expr::QLit { .. } => false,
            Expr::BitLiteral { dim, .. } => *dim == 1,
            Expr::QubitRef { .. } => true,
        }
    }

    pub fn eval_step(&self, state: &mut ReplState) -> Option<Expr> {
        match self {
            Expr::QLit { qlit, .. } => match qlit {
                QLit::ZeroQubit { .. } => {
                    let q = state.sim.allocate();
                    Some(Expr::QubitRef { index: q })
                }
                QLit::OneQubit { .. } => {
                    let q = state.sim.allocate();
                    state.sim.x(q); // use x to flip 0 to 1
                    Some(Expr::QubitRef { index: q })
                }
                QLit::QubitTilt { q, angle_deg, dbg } => {
                    let inside_expr = Expr::QLit {
                        qlit: *q.clone(),
                        dbg: dbg.clone(),
                    }; // recursion for nest
                    if let Some(Expr::QubitRef { index }) = inside_expr.eval_step(state) {
                        state.sim.rz(*angle_deg, index); // tilt using spacesim
                        Some(Expr::QubitRef { index })
                    } else {
                        None // evaluation failed
                    }
                }
                QLit::UniformSuperpos { q1, q2, dbg } => {
                    let inside_expr = Expr::QLit {
                        qlit: *q1.clone(),
                        dbg: dbg.clone(),
                    };
                    if let Some(Expr::QubitRef { index }) = inside_expr.eval_step(state) {
                        state.sim.h(index);
                        Some(Expr::QubitRef { index })
                    } else {
                        None
                    }
                }
                // qs is a vector, so parse through vector and then evaluate
                QLit::QubitTensor { qs, dbg } => {
                    let mut vals = Vec::new();
                    for qlit in qs {
                        let inner_expr = Expr::QLit {
                            qlit: qlit.clone(),
                            dbg: dbg.clone(),
                        };
                        if let Some(Expr::QubitRef { index }) = inner_expr.eval_step(state) {
                            vals.push(Expr::QubitRef { index });
                        } else {
                            return None;
                        }
                    }
                    Some(Expr::Tensor {
                        vals,
                        dbg: dbg.clone(),
                    })
                }
                QLit::QubitUnit { dbg, .. } => Some(Expr::UnitLiteral { dbg: dbg.clone() }),
            },
            Expr::QubitRef { .. } => None,
            _ => todo!("eval_step()"),
        }
    }

    pub fn eval_to_value(&self, state: &mut ReplState) -> Expr {
        let mut expr = self.clone();
        loop {
            match expr.eval_step(state) {
                Some(new_expr) => {
                    expr = new_expr;
                }
                None => {
                    return expr;
                }
            }
        }
    }
}
