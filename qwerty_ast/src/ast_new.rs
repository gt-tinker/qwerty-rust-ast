// ================================================
// 
// ================================================

#[derive(Debug, Clone)]
pub enum Type {
    FuncType { in_ty: Box<Type>, out_ty: Box<Type> },
    RevFuncType { in_out_ty: Box<Type> },
    RegType { elem_ty: RegKind, dim: u64 },
    UnitType,
}

#[derive(Debug, Clone)]
pub enum RegKind {
    Bit,
    Qubit,
    Basis,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Variable { name: String },
    UnitLiteral,
    Adjoint { func: Box<Expr> },
    Pipe { lhs: Box<Expr>, rhs: Box<Expr> },
    Measure { b: Basis },
    Discard,
    Tensor { vals: Vec<Expr> },
    BasisTranslation { bin: Basis, bout: Basis },
    Predicated { then_func: Box<Expr>, else_func: Box<Expr>, pred: Basis },
    NonUniformSuperpos { pairs: Vec<(f64, Qlit)> },
    Conditional { then_branch: Box<Expr>, else_branch: Box<Expr>, cond: Box<Expr> },
    Qlit { qlit: Qlit },
}

#[derive(Debug, Clone)]
pub enum Qlit {
    ZeroQubit,
    OneQubit,
    QubitTilt { q: Box<Qlit> },
    UniformSuperpos { q1: Box<Qlit>, q2: Box<Qlit> },
    QubitTensor { qs: Vec<Qlit> },
}

#[derive(Debug, Clone)]
pub enum VecLit {
    ZeroVector,
    OneVector,
    PadVector,
    TargetVector,
    VectorTilt { q: Box<Qlit> },
    UniformVectorSuperpos { q1: Box<Qlit>, q2: Box<Qlit> },
    VectorTensor { qs: Vec<Qlit> },
}

#[derive(Debug, Clone)]
pub enum Basis {
    BasisLiteral { vecs: Vec<VecLit> },
    EmptyBasisLiteral,
    BasisTensor { bases: Vec<Basis> },
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Assign { lhs: String, rhs: Expr },
    UnpackAssign { lhs: Vec<String>, rhs: Expr },
    Return { val: Expr },
}

#[derive(Debug, Clone)]
pub struct FuncArg {
    pub ty: Type,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct Func {
    pub name: String,
    pub args: Vec<FuncArg>,
    pub ret_type: Type,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub struct Prog {
    pub funcs: Vec<Func>,
}
