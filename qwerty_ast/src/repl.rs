//! This module holds the state of evaluation in the Qwerty REPL and how steps
//! of evaluation are taken. The latter is based loosely on Appendix A of
//! arXiv:2404.12603.

use crate::ast::{
    Adjoint, BitLiteral, Conditional, Expr, Predicated, QLit, QubitRef, Stmt, Tensor, UnitLiteral,
};
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
        if let Stmt::Expr(expr) = stmt {
            expr.eval_to_value(self)
        } else {
            Expr::UnitLiteral(UnitLiteral { dbg: None })
        }
    }
}

impl QLit {
    pub fn eval_step(&self, state: &mut ReplState) -> Option<Expr> {
        match self {
            QLit::ZeroQubit { .. } => {
                let index = state.sim.allocate();
                Some(Expr::QubitRef(QubitRef { index }))
            }
            QLit::OneQubit { .. } => {
                let index = state.sim.allocate();
                state.sim.x(index); // use x to flip 0 to 1
                Some(Expr::QubitRef(QubitRef { index }))
            }
            QLit::QubitTilt { q, angle_deg, .. } => {
                let inside_expr = q.eval_step(state)?;
                if let Expr::QubitRef(QubitRef { index }) = inside_expr {
                    state.sim.rz(*angle_deg, index); // tilt using spacesim
                    Some(Expr::QubitRef(QubitRef { index }))
                } else {
                    None // evaluation failed
                }
            }
            QLit::UniformSuperpos { .. } => todo!("UniformSuperpos"),
            // qs is a vector, so parse through vector and then evaluate
            QLit::QubitTensor { qs, dbg } => {
                let mut vals = Vec::new();
                for qlit in qs {
                    let inner_expr = qlit.eval_step(state)?;
                    match inner_expr {
                        Expr::QubitRef(_) => vals.push(inner_expr),
                        _ => return None,
                    }
                }
                Some(Expr::Tensor(Tensor {
                    vals,
                    dbg: dbg.clone(),
                }))
            }
            QLit::QubitUnit { dbg } => Some(Expr::UnitLiteral(UnitLiteral { dbg: dbg.clone() })),
        }
    }
}

impl Expr {
    pub fn is_value(&self) -> bool {
        match self {
            Expr::Variable(_) => false,
            Expr::UnitLiteral(_) => true,
            Expr::Adjoint(Adjoint { func, .. }) => func.is_value(),
            Expr::Pipe(_) => false,
            Expr::Measure(_) => true,
            Expr::Discard(_) => true,
            Expr::Tensor(Tensor { vals, .. }) => vals
                .iter()
                .all(|v| v.is_value() && !matches!(v, Expr::UnitLiteral { .. })),
            Expr::BasisTranslation(_) => true,
            Expr::Predicated(Predicated {
                then_func,
                else_func,
                ..
            }) => then_func.is_value() && else_func.is_value(),
            Expr::NonUniformSuperpos(_) => false,
            Expr::Conditional(Conditional {
                then_expr,
                else_expr,
                cond,
                ..
            }) => then_expr.is_value() && else_expr.is_value() && cond.is_value(),
            Expr::QLit(_) => false,
            Expr::BitLiteral(BitLiteral { dim, .. }) => *dim == 1,
            Expr::QubitRef(_) => true,
        }
    }

    pub fn eval_step(&self, state: &mut ReplState) -> Option<Expr> {
        match self {
            Expr::QLit(qlit) => qlit.eval_step(state),
            Expr::QubitRef(_) | Expr::UnitLiteral(_) => None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::angle_is_approx_zero;
    use num_complex::Complex64;

    fn assert_state_vectors_are_approx_equal(expected: &[Complex64], actual: &[Complex64]) {
        assert_eq!(expected.len(), actual.len());
        assert!(expected
            .iter()
            .zip(actual.iter())
            .all(|(expected_amp, actual_amp)| angle_is_approx_zero(
                (expected_amp - actual_amp).norm()
            )));
    }

    // '0' -> value: q[0]
    //        state vector: [1 0]
    #[test]
    fn test_eval_qlit_zero() {
        let mut repl_state = ReplState::new();
        let start_expr = Expr::QLit(QLit::ZeroQubit { dbg: None });

        let actual_end_expr = start_expr.eval_to_value(&mut repl_state);
        let expected_end_expr = Expr::QubitRef(QubitRef { index: 0 });
        assert_eq!(expected_end_expr, actual_end_expr);

        let actual_end_state = repl_state.sim.get_state_vector();
        let expected_end_state = vec![Complex64::ONE, Complex64::ZERO];
        assert_state_vectors_are_approx_equal(&expected_end_state, &actual_end_state);
    }

    // '1' -> value: q[0]
    //        state vector: [0 1]
    #[test]
    fn test_eval_qlit_one() {
        let mut repl_state = ReplState::new();
        let start_expr = Expr::QLit(QLit::OneQubit { dbg: None });

        let actual_end_expr = start_expr.eval_to_value(&mut repl_state);
        let expected_end_expr = Expr::QubitRef(QubitRef { index: 0 });
        assert_eq!(expected_end_expr, actual_end_expr);

        let actual_end_state = repl_state.sim.get_state_vector();
        let expected_end_state = vec![Complex64::ZERO, Complex64::ONE];
        assert_state_vectors_are_approx_equal(&expected_end_state, &actual_end_state);
    }

    // '0'*'1' -> value: q[0]*q[1]
    //            state vector: [0 0 1 0]
    #[test]
    fn test_eval_qlit_tensor_zero_one() {
        let mut repl_state = ReplState::new();
        let start_expr = Expr::QLit(QLit::QubitTensor {
            qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None }],
            dbg: None,
        });

        let actual_end_expr = start_expr.eval_to_value(&mut repl_state);
        let expected_end_expr = Expr::Tensor(Tensor {
            vals: vec![
                Expr::QubitRef(QubitRef { index: 0 }),
                Expr::QubitRef(QubitRef { index: 1 }),
            ],
            dbg: None,
        });
        assert_eq!(expected_end_expr, actual_end_expr);

        let actual_end_state = repl_state.sim.get_state_vector();
        let expected_end_state = vec![
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ONE,
            Complex64::ZERO,
        ];
        assert_state_vectors_are_approx_equal(&expected_end_state, &actual_end_state);
    }
}
