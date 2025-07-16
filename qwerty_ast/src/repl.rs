use std::collections::HashMap;
use quantum_sparse_sim::QuantumSim;
use crate::ast::Expr;

#[derive(Clone)]

pub struct ReplState {
    // sim: QuantumSim,
    bindings: HashMap<String, Expr>
    // TODO figure out what expr should be
}

impl ReplState {
    pub fn new() -> Self {
        // ReplState { sim: QuantumSim::new(None), bindings: HashMap::new() }
        ReplState {bindings: HashMap::new()}
    }
    
    pub fn run(&self, expr: &Expr) {
        // TODO run
        println!("Running {:?}", expr);
    }
}

