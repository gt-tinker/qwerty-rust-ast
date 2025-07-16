//! Wraps qwerty_ast::repl::ReplState in a Python object. Used by repl.py to
//! run the Qwerty REPL.

use crate::wrap_ast::Expr;
use pyo3::prelude::*;
use qwerty_ast::repl;
use std::sync::Mutex;

#[pyclass]
pub struct ReplState {
    // Mutex used here because PyO3 requires #[pyclass]es to be Sync, i.e.,
    // threadsafe, but ReplState is not Sync because it contains QuantumSim
    // which contains a non-Sync RefCell.
    state: Mutex<repl::ReplState>,
}

#[pymethods]
impl ReplState {
    #[new]
    fn new() -> Self {
        Self {
            state: Mutex::new(repl::ReplState::new()),
        }
    }

    fn run(&self, expr: Expr) {
        self.state.lock().unwrap().run(&expr.expr)
    }
}
