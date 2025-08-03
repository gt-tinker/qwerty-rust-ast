//! This module holds the state of evaluation in the Qwerty REPL and how steps
//! of evaluation are taken. The latter is based loosely on Appendix A of
//! arXiv:2404.12603.

use crate::ast::{
    Adjoint, BitExpr, Conditional, EmbedClassical, Predicated, QLit, QpuExpr, QubitRef, Stmt,
    Tensor, UnitLiteral,
};
use quantum_sparse_sim::QuantumSim;
use std::collections::HashMap;

/// Holds the quantum simulator state and a mapping of names to values.
pub struct ReplState {
    sim: QuantumSim,
    // bindings: HashMap<String, Expr>, // TODO: figure out what expr should be
    // Bindings now store QpuExpr, as the REPL evaluates QPU expressions. right?
    bindings: HashMap<String, QpuExpr>,
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
    // Stmt now requires a generic parameter. Assuming QpuExpr for REPL context.
    pub fn run(&mut self, stmt: &Stmt<QpuExpr>) -> QpuExpr {
        if let Stmt::Expr(stmt_expr) = stmt {
            stmt_expr.expr.eval_to_value(self)
        } else {
            QpuExpr::UnitLiteral(UnitLiteral { dbg: None })
        }
    }
}

impl QpuExpr {
    pub fn is_value(&self) -> bool {
        match self {
            QpuExpr::Variable(_) => false,
            QpuExpr::UnitLiteral(_) => true,
            QpuExpr::EmbedClassical(EmbedClassical { func, .. }) => func.is_value(),
            QpuExpr::Adjoint(Adjoint { func, .. }) => func.as_ref().is_value(),
            QpuExpr::Pipe(_) => false,
            QpuExpr::Measure(_) => true,
            QpuExpr::Discard(_) => true,
            QpuExpr::Tensor(Tensor { vals, .. }) => vals
                .iter()
                .all(|v| v.is_value() && !matches!(v, QpuExpr::UnitLiteral(_))),
            QpuExpr::BasisTranslation(_) => true,
            QpuExpr::Predicated(Predicated {
                then_func,
                else_func,
                ..
            }) => then_func.as_ref().is_value() && else_func.as_ref().is_value(),
            QpuExpr::NonUniformSuperpos(_) => false,
            QpuExpr::Conditional(Conditional {
                then_expr,
                else_expr,
                cond,
                ..
            }) => {
                // then_expr and else_expr are QpuExpr, cond is BitExpr
                then_expr.as_ref().is_value()
                    && else_expr.as_ref().is_value()
                    && cond.as_ref().is_value()
            }
            QpuExpr::QLit(_) => false,
            // Removed Expr::BitLiteral as it's not part of QpuExpr grammar (TODO: Confirm with Austin!)
            QpuExpr::QubitRef(_) => true,
        }
    }

    pub fn eval_step(&self, state: &mut ReplState) -> Option<QpuExpr> {
        match self {
            QpuExpr::QLit(qlit) => match qlit {
                QLit::ZeroQubit { .. } => Some(QpuExpr::QubitRef(QubitRef {
                    index: state.sim.allocate(),
                })),
                _ => todo!("Rest of QLit eval_step"),
            },
            QpuExpr::QubitRef { .. } => None,
            _ => todo!("eval_step() for QpuExpr"),
        }
    }

    pub fn eval_to_value(&self, state: &mut ReplState) -> QpuExpr {
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

impl BitExpr {
    pub fn is_value(&self) -> bool {
        match self {
            BitExpr::Variable(_) => false,
            BitExpr::BitLiteral(_) => true,
            _ => false,
        }
    }

    pub fn eval_step(&self, _state: &mut ReplState) -> Option<BitExpr> {
        // TODO: Implement evaluation steps for classical expressions here
        None
    }

    pub fn eval_to_value(&self, state: &mut ReplState) -> BitExpr {
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
