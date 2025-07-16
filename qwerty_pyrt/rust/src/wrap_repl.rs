use pyo3::prelude::*;
use qwerty_ast::repl;
use crate::wrap_ast::Expr;
// use std::sync::Mutex;

#[pyclass]
#[derive(Clone)]
pub struct ReplState {
    // state: Mutex<repl::ReplState>
    state: repl::ReplState
}

#[pymethods]
impl ReplState {
    #[new]
    fn new() -> Self {
        Self {
            state: repl::ReplState::new(),
        }
    }

    fn run(&self, expr: Expr) {
        // TODO run
        self.state.run(&expr.expr)
    }
}