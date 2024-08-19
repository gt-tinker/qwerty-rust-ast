#![allow(unused_imports, dead_code)]
/// This file will define all of the necessary functions for
/// type inference (HM type inference)
// FIXME: Remove later!
use crate::ast::*;
use crate::dimexpr::{BinOp, DimExpr, DimVar};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use std::collections::{HashMap, HashSet};

// The global type context
pub type Context = HashMap<String, Type>;

// LHS is equal to RHS
pub type DimConstraint = (DimExpr, DimExpr);

#[derive(Clone, Debug, Default)]
pub struct ConstraintInfo {
    pub dimexprs: HashSet<DimExpr>,
    pub dimvars: HashMap<DimVar, DimExpr>,
    pub constraints: Vec<DimConstraint>,
}

impl ConstraintInfo {
    pub fn new() -> Self {
        Self {
            dimexprs: HashSet::new(),
            dimvars: HashMap::new(),
            constraints: Vec::new(),
        }
    }

    pub fn contains(&self, dimvar: &DimVar) -> bool {
        self.dimvars.contains_key(dimvar)
    }

    // Updates the contents of the ConstraintInfo
    // with info from a Dimension Expression
    pub fn update(&mut self, dimexpr: Option<DimExpr>) {
        if dimexpr.is_none() {
            return;
        }

        fn update_dimvars(dimexpr: &DimExpr, dimvars: &mut HashMap<DimVar, DimExpr>) {
            match dimexpr {
                DimExpr::Variable(val) => {
                    dimvars.insert(val.to_string(), DimExpr::Number(0)); // Default value
                }
                DimExpr::Number(_) => {}
                DimExpr::Unary(_, expr) => update_dimvars(expr, dimvars),
                DimExpr::Binary(_, expr1, expr2) => {
                    update_dimvars(expr1, dimvars);
                    update_dimvars(expr2, dimvars);
                }
            };
        }
        let dimexpr = dimexpr.unwrap();
        update_dimvars(&dimexpr, &mut self.dimvars);

        self.dimexprs.insert(dimexpr);
    }

    // Using `dimexprs`, generate constraints
    pub fn generate_constraints(&mut self) {
        let mut new_constraints: Vec<DimConstraint> = Vec::new();
        for (i, de1) in self.dimexprs.iter().enumerate() {
            for (j, de2) in self.dimexprs.iter().enumerate() {
                if i < j {
                    new_constraints.push((de1.clone(), de2.clone()));
                }
            }
        }
        self.constraints.extend(new_constraints);
    }

    // TODO: Solver function!
    // pub fn solve(&mut self) {
    //     // Simplify
    //     for (lhs, rhs) in self.constraints.iter_mut() {
    //         lhs.simplify_with_passes();
    //         rhs.simplify_with_passes();
    //         dbg!(lhs, rhs);
    //
    //         // After simplifying, and symmetry (to get all variables to one side, all
    //         // constants to the other), we can parse the tree to get coeffs
    //         // for the variables
    //     }
    //
    //     // create a function that uses the updated constraints to create the matrix
    //
    //     // Then do symmetry and populate the matrix
    //     for (lhs, rhs) in self.constraints.iter() {
    //         let constraint_row = lhs.symmetry(&rhs);
    //     }
    // }

    pub fn solve(&mut self) {
        // Simplify all constraints
        for (lhs, rhs) in self.constraints.iter_mut() {
            lhs.simplify_with_passes();
            rhs.simplify_with_passes();
        }

        // Number of constraints and variables
        let num_constraints = self.constraints.len();
        let num_vars = self.dimvars.len();

        // Create a map from variable names to column indices
        let mut var_indices: Vec<&String> = self.dimvars.keys().collect();
        var_indices.sort(); // Sort alphabetically for consistency

        // Initialize matrix A and vector b
        let mut a_matrix = DMatrix::zeros(num_constraints, num_vars);
        let mut b_vector = DVector::zeros(num_constraints);

        // Populate matrix A and vector b
        for (row_idx, (lhs, rhs)) in self.constraints.iter().enumerate() {
            // Generate the combined constraint
            let constraint = lhs.symmetry(rhs);

            // Populate the b vector with the sum of constants
            b_vector[row_idx] = constraint.sum_consts as f64;

            // Populate the A matrix with coefficients for each variable
            for var_info in &constraint.vars {
                if let Some(col_idx) = var_indices.iter().position(|&var| var == &var_info.name) {
                    a_matrix[(row_idx, col_idx)] = var_info.coeff as f64;
                }
            }
        }

        // TODO: Solve the system Ax = b (if solvable)
        // Example: Use QR decomposition or other nalgebra solving techniques
        // Solve using SVD decomposition
        let svd = a_matrix.svd(true, true); // Perform SVD decomposition
        if let Ok(solution) = svd.solve(&b_vector, 1e-6) {
            // 1e-6 is the tolerance for singular values
            // Map the solution to dimension variables
            for ((_, val), value) in self.dimvars.iter_mut().zip(solution.iter()) {
                *val = DimExpr::Number(*value as isize);
            }
        } else {
            println!("No solution found.");
        }
        dbg!(&self.dimvars);
    }
}

// FIXME: Add ConstraintInfo tests for solving

#[derive(Clone, Debug, PartialEq)]
pub struct DimVarInfo {
    pub name: String,
    pub op: BinOp,
    pub coeff: isize,
}

impl std::ops::Sub for DimVarInfo {
    type Output = Option<Self>;

    fn sub(self, rhs: Self) -> Self::Output {
        // Ensure we're subtracting the same variable
        if self.name != rhs.name {
            return None; // Cannot subtract different variables
        }

        // Handle the operations
        let self_coeff = self.coeff as f64;
        let rhs_coeff = match rhs.op {
            BinOp::Mul => rhs.coeff as f64,
            BinOp::Div => 1.0 / rhs.coeff as f64,
            _ => return None, // Unsupported operation for subtraction
        };

        let new_coeff = match self.op {
            BinOp::Mul => self_coeff - rhs_coeff,
            BinOp::Div => self_coeff - (1.0 / rhs_coeff), // Handle Div on LHS
            _ => return None,                             // Unsupported operation for subtraction
        };

        if new_coeff == 0.0 {
            return None; // Resultant coefficient is zero, term cancels out
        }

        // Return the result as a new DimVarInfo with integer coefficient
        Some(Self {
            name: self.name,
            op: BinOp::Mul,                    // Result always expressed as Mul
            coeff: new_coeff.floor() as isize, // Integer division result
        })
    }
}

#[test]
fn test_sub_dimvarinfo() {
    let lhs = DimVarInfo {
        name: "N".to_string(),
        op: BinOp::Mul,
        coeff: 2,
    };

    let rhs = DimVarInfo {
        name: "N".to_string(),
        op: BinOp::Div,
        coeff: 2,
    };

    let res = lhs - rhs;
    let ans = DimVarInfo {
        name: "N".to_string(),
        op: BinOp::Mul,
        coeff: 1,
    };
    assert_eq!(res.unwrap(), ans);
}

#[derive(Clone, Debug, Default)]
pub struct SimplifiedConstraint {
    pub sum_consts: isize,
    pub vars: Vec<DimVarInfo>,
}

impl SimplifiedConstraint {
    pub fn normalize_constraints(lhs: &Self, rhs: &Self) -> Self {
        // Start with a copy of LHS
        let mut normalized = lhs.clone();

        // Subtract constants
        normalized.sum_consts = rhs.sum_consts - lhs.sum_consts;

        // Prepare to subtract variables
        let mut rhs_vars = rhs.vars.clone();

        // For each LHS variable, attempt to find a match in RHS and subtract
        normalized.vars = normalized
            .vars
            .into_iter()
            .filter_map(|lhs_var| {
                if let Some(pos) = rhs_vars
                    .iter()
                    .position(|rhs_var| rhs_var.name == lhs_var.name && rhs_var.op == lhs_var.op)
                {
                    let rhs_var = rhs_vars.remove(pos);
                    lhs_var - rhs_var // Subtract the matching variable
                } else {
                    Some(lhs_var) // Keep unmatched LHS variable as-is
                }
            })
            .collect();

        // Add remaining RHS variables as negative contributions
        for rhs_var in rhs_vars {
            normalized.vars.push(DimVarInfo {
                name: rhs_var.name,
                op: rhs_var.op.clone(),
                coeff: -rhs_var.coeff,
            });
        }

        normalized
    }
}

// Substitution is a mapping from string to type
// NOTE: Should Substitution be ordered?
// Goes from TypeVar to Type, so we use uid as usize
pub type Substitution = HashMap<usize, Type>;

// Add types for DimSubstitution?

fn instantiate_global_map() -> Context {
    let mut ctx: HashMap<String, Type> = HashMap::new();

    // All of the qubit symbols have a defined type of Qubit[1]
    // NOTE: Is this necessary?
    // ctx.insert(
    //     "p".to_string(),
    //     Type::RegisterType(RegisterType {
    //         variant: RegisterVariant::Qubit,
    //         dim: ast_dim_const![1],
    //     }),
    // );

    // NOTE: We should also add defined functions such as the built-in bases,
    // flip, id, sign, etc., later

    ctx
}

/// This function does magic
/// Assume the first node passed in is the Program node
pub fn infer(node: &mut Program) {
    let mut tyvar_generator = TypeVarGenerator::new();
    let mut root_ctx = instantiate_global_map();
    for child in &mut node.body {
        infer_helper(child, &mut tyvar_generator, &mut root_ctx);
    }

    // dbg!(root_ctx);
}

// FIXME: Should infer_helper return the type? Or just substitutions?
fn infer_helper(
    node: &mut ASTNode,
    tyvar_generator: &mut TypeVarGenerator,
    ctx: &mut Context,
) -> (Substitution, Option<ConstraintInfo>) {
    // TODO: For now, just try creating a function that
    // changes types of every node to UnitType

    // FIXME: Remove this at some point
    let unit_type: Option<Type> = Some(Type::UnitType());

    // Do a little trolling here
    match node {
        ASTNode::Return(ret) => {
            // Recursively update stuff
            let (in_ret_subst, constraint_info) =
                infer_helper(&mut ret.value, tyvar_generator, ctx);
            let ret_typ = ret.value.get_typ().unwrap(); // FIXME: Remove the unwrap later
            let tyvar_ret_type = tyvar_generator.new_tyvar();
            let mut new_subst = unify(
                ctx,
                ret_typ.clone(),
                Type::TypeVar(tyvar_ret_type),
                Substitution::new(),
            );
            compose(&mut new_subst, in_ret_subst);
            // NOTE: To Self: we have compose, to extend substitutions
            ret.typ = Some(ret_typ);
            (new_subst, constraint_info)
        }
        ASTNode::Pipe(pipe) => {
            // 1. Get TyVar for "return type" of Pipe, since Pipe is
            //    basically a function call on an input
            let pipe_return_type = tyvar_generator.new_tyvar();
            // 2. Update pipe.typ with temp tyvar type
            pipe.typ = Some(Type::TypeVar(pipe_return_type.clone()));
            // 3. Compose LHS and RHS substitutions
            // FIXME: Add to func_constraint_info with type information
            let (mut lhs_subst, constraint_info) =
                infer_helper(&mut pipe.lhs, tyvar_generator, ctx);
            let mut constraint_info = constraint_info.unwrap();
            let (rhs_subst, other_constraint_info) =
                infer_helper(&mut pipe.rhs, tyvar_generator, ctx);
            compose(&mut lhs_subst, rhs_subst);
            compose_constraint_info(&mut constraint_info, other_constraint_info);
            // 4. Get "ret_typ" (rhs of rhs' type) and unify with ret_tyvar
            //    using the substitutions
            let lhs_type = pipe.lhs.get_typ().unwrap();
            let rhs_type = pipe.rhs.get_typ().unwrap(); // FuncType

            let lhs_dim = pipe.lhs.get_dim();
            constraint_info.update(lhs_dim);
            if let Type::FuncType(ref func) = rhs_type {
                // NOTE: Unsure if these are both necessary
                let func_lhs_dim = func.lhs.get_dim();
                let func_rhs_dim = func.rhs.get_dim();

                constraint_info.update(func_lhs_dim);
                constraint_info.update(func_rhs_dim);
            }

            let pipe_func_type = Type::FuncType(FuncType {
                lhs: Box::new(lhs_type.clone()),
                rhs: Box::new(Type::TypeVar(pipe_return_type.clone())),
                is_rev: false,
            });

            dbg!(&pipe_func_type, &rhs_type);

            let func_unified_subst = unify(ctx, pipe_func_type, rhs_type, lhs_subst.clone());

            dbg!(&func_unified_subst);

            compose(&mut lhs_subst, func_unified_subst);

            // 5. Use `apply_ty` on the pipe_type with substitutions, then
            //    do same with overall context
            apply_ty(pipe.typ.as_mut().unwrap(), &lhs_subst);
            apply_ctx(ctx, &lhs_subst);

            (lhs_subst, Some(constraint_info))
        }
        ASTNode::Basis(basis) => {
            // Go through all basis elements, sum up dims
            let mut dim = basis.dim.clone();
            dim.simplify();
            basis.typ = Some(Type::BasisType(BasisType { dim }));
            (Substitution::new(), Some(ConstraintInfo::default()))
        }
        ASTNode::BasisTranslation(_bt) => {
            // Go through all basis elements, sum up dims
            // let mut dim = bt.dim.clone();
            // dim.simplify();
            // bt.typ = Some(Type::BasisType(BasisType { dim }));
            // FIXME: Add handling for types as well as dimension constraints, similar to Pipe
            (Substitution::new(), Some(ConstraintInfo::default()))
        }
        ASTNode::BuiltInBasis(basis) => {
            basis.typ = unit_type;
            (Substitution::new(), Some(ConstraintInfo::default()))
        }
        ASTNode::Measure(measure) => {
            // type of measure is Qubit[N] -> Bit[N]
            let (subst, constraint_info) = infer_helper(&mut measure.basis, tyvar_generator, ctx);
            let mut basis_dim = measure.basis.get_dim().unwrap();
            basis_dim.simplify();
            let qubit_reg_type = Type::RegisterType(RegisterType {
                variant: RegisterVariant::Qubit,
                dim: basis_dim.clone(),
            });
            let bit_reg_type = Type::RegisterType(RegisterType {
                variant: RegisterVariant::Bit,
                dim: basis_dim,
            });
            measure.typ = Some(Type::FuncType(FuncType {
                lhs: Box::new(qubit_reg_type),
                rhs: Box::new(bit_reg_type),
                is_rev: false,
            }));
            (subst, constraint_info)
        }
        ASTNode::Pred(pred) => {
            let (mut subst, constraint_info) = infer_helper(&mut pred.basis, tyvar_generator, ctx);
            let mut constraint_info = constraint_info.unwrap();
            let (body_subst, body_constraint_info) =
                infer_helper(&mut pred.body, tyvar_generator, ctx);
            compose(&mut subst, body_subst);
            compose_constraint_info(&mut constraint_info, body_constraint_info);

            let basis_dim = pred.basis.get_dim().unwrap();
            let body_dim = pred.body.get_dim().unwrap();
            let mut pred_dim = basis_dim + body_dim;
            pred_dim.simplify();
            let reg_type = Type::RegisterType(RegisterType {
                variant: RegisterVariant::Qubit,
                dim: pred_dim,
            });

            pred.typ = Some(Type::FuncType(FuncType {
                lhs: Box::new(reg_type.clone()),
                rhs: Box::new(reg_type),
                is_rev: true,
            }));

            (subst, Some(constraint_info))
        }
        ASTNode::BasisLiteral(bl) => {
            bl.typ = unit_type;
            (Substitution::new(), Some(ConstraintInfo::default()))
        }
        ASTNode::Flip(flip) => {
            let (subst, constraint_info) = infer_helper(&mut flip.basis, tyvar_generator, ctx);
            // let basis_typ = flip.basis.get_typ().unwrap();
            // Flip is a reversible function
            // from Qubit[N] - rev -> Qubit[N]
            // NOTE: Steps:
            // 1. Get dim from basis
            // TODO: Fix with custom errors later!
            let mut dim = flip.basis.get_dim().unwrap();
            dim.simplify();
            // 2. Create register type with dim
            let reg_type = Type::RegisterType(RegisterType {
                variant: RegisterVariant::Qubit,
                dim,
            });
            // 3. Create function type with register types
            let func_type = Type::FuncType(FuncType {
                lhs: Box::new(reg_type.clone()),
                rhs: Box::new(reg_type),
                is_rev: true,
            });
            // 4. Profit
            flip.typ = Some(func_type);
            (subst, constraint_info)
        }
        ASTNode::Identity(id) => {
            // Get dim, then update type
            let mut dim = id.dim.clone();
            dim.simplify();
            let reg_type = Type::RegisterType(RegisterType {
                variant: RegisterVariant::Qubit,
                dim,
            });

            id.typ = Some(Type::FuncType(FuncType {
                lhs: Box::new(reg_type.clone()),
                rhs: Box::new(reg_type),
                is_rev: true,
            }));
            (Substitution::new(), Some(ConstraintInfo::default()))
        }
        ASTNode::QubitLiteral(ql) => {
            // TODO: Fix this so we can have correct multiplication between factor and elts len
            let mut dim = ql.factor.clone() * DimExpr::Number(ql.elts.len() as isize);
            dim.simplify();
            ql.typ = Some(Type::RegisterType(RegisterType {
                variant: RegisterVariant::Qubit,
                // FIXME: See if we can remove clone!
                dim,
            }));
            (Substitution::new(), Some(ConstraintInfo::default()))
        }
        ASTNode::Tensor(_tensor) => {
            // TODO: pls fix
            // Need to combine the dimension exprs (add together)
            // Goals:
            // 1. Get type of first element
            // 2. Verify that types of all other elements match (dimensions aside)
            // 3. Add all dimensions together
            // 4. Multiply by factor of tensor node
            (Substitution::new(), Some(ConstraintInfo::default()))
        }
        ASTNode::Program(_) => {
            // Handled in `infer` function itself
            unreachable!()
        }
        ASTNode::Function(func) => {
            // FIXME: Currently only supports 0 or 1 argument (and return val) to the function
            let param_type = if func.args.len() == 0 {
                Type::UnitType()
            } else {
                Type::TypeVar(tyvar_generator.new_tyvar())
            };

            let ret_type = Type::TypeVar(tyvar_generator.new_tyvar());

            let func_type = Type::FuncType(FuncType {
                lhs: Box::new(param_type),
                rhs: Box::new(ret_type.clone()),
                is_rev: false, // How do we do this dynamically?
            });

            // NOTE: Do we do the updates before or after the application of
            // substitutions?

            // add locally for now
            // FIXME: Do we need this?
            func.typ = Some(func_type.clone());

            // update in env
            ctx.insert(func.name.clone(), func_type);

            // Verify that last node in body is return, else not well typed

            let mut func_subst = Substitution::new();
            // Create constraint_info here?
            let mut constraint_info = ConstraintInfo::default();
            let len = func.body.len();
            for (i, stmt) in func.body.iter_mut().enumerate() {
                let (stmt_subst, stmt_constraint_info) = infer_helper(stmt, tyvar_generator, ctx);
                // compose
                compose(&mut func_subst, stmt_subst);
                compose_constraint_info(&mut constraint_info, stmt_constraint_info);

                if let ASTNode::Return(ret) = stmt {
                    if i == len - 1 {
                        let ret_node_typ = ret.value.get_typ().unwrap();
                        let unify_subst =
                            unify(ctx, ret_type.clone(), ret_node_typ, func_subst.clone());
                        compose(&mut func_subst, unify_subst);
                    } else {
                        panic!("Return was not last node of function, found instead.");
                    }
                }
            }
            // Apply substitutions to the function type and the context
            apply_ty(func.typ.as_mut().unwrap(), &func_subst);
            apply_ctx(ctx, &func_subst);

            constraint_info.generate_constraints();
            // FIXME: Uncomment this later!
            // constraint_info.solve();
            func.constraint_info = constraint_info;

            // FIXME: We need to propagate this information
            // through the types of all of the nodes!
            // (This is tedious bruh)

            // We aren't returning anything up the chain, since we're
            // at the function we care about
            (func_subst, None)
        }
        // TODO: Variable, Call, Assign, Basis (for teleport.py)
        ASTNode::Variable(var) => {
            // FIXME: Fill this out!
            (Substitution::new(), Some(ConstraintInfo::default()))
        }
        ASTNode::Assign(assign) => {
            // FIXME: Fill this out!
            (Substitution::new(), Some(ConstraintInfo::default()))
        }
        ASTNode::Call(call) => {
            // FIXME: Fill this out!
            (Substitution::new(), Some(ConstraintInfo::default()))
        }
        ASTNode::Basis(basis) => {
            // FIXME: Fill this out!
            (Substitution::new(), Some(ConstraintInfo::default()))
        }
        other => (Substitution::new(), Some(ConstraintInfo::default())),
    }
}

// Produces the substitution that binds ty1 to ty2
// FIXME: Add a field for a pre-existing substitution to consider
// in the current unification? Perhaps, if we are using two typevars?
fn unify(ctx: &mut Context, ty1: Type, ty2: Type, subst_in: Substitution) -> Substitution {
    // FIXME: We need to use unify to handle the dim var constraint
    // generation, perhaps add helper functions?

    let mut subst = Substitution::new();
    compose(&mut subst, subst_in);
    // perform occurs check
    match (ty1, ty2) {
        (Type::TypeVar(var), other) | (other, Type::TypeVar(var)) => {
            if occurs(&var, &other) {
                panic!("Type variable occurs in other type, cannot perform unification");
            }
            // Bind variable to type
            subst.insert(var.uid, other);
            return subst;
        }
        (Type::FuncType(f1), Type::FuncType(f2)) => {
            let lhs_subst = unify(ctx, *f1.lhs, *f2.lhs, Substitution::new());
            let rhs_subst = unify(ctx, *f1.rhs, *f2.rhs, Substitution::new());
            compose(&mut subst, lhs_subst);
            compose(&mut subst, rhs_subst);
            return subst;
        }
        // TODO: Check if both ty1, ty2 are functions, check args, outs, etc.
        _ => {}
    }
    subst
}

// FIXME: Do we need this?
fn apply_ctx(ctx: &mut Context, subst: &Substitution) {
    for typ in ctx.values_mut() {
        apply_ty(typ, subst);
    }
}

// FIXME: Do we need this?
fn apply_ty(ty: &mut Type, subst: &Substitution) {
    match ty {
        Type::TypeVar(var) => {
            if let Some(new_ty) = subst.get(&var.uid) {
                *ty = new_ty.clone();
            }
        }
        Type::FuncType(func) => {
            apply_ty(&mut func.lhs, subst);
            apply_ty(&mut func.rhs, subst);
        }
        // Add cases for other Type variants if needed
        _ => {}
    }
}

/// Given a type variable `tyvar`, swap all instances of it
/// with the type `to_typ` in the type `in_typ`
// FIXME: Add more stuff here! Unsure if necessary
fn subst_var_typ(tyvar: TypeVar, to_typ: Type, in_typ: &mut Type) {
    match in_typ {
        Type::TypeVar(var) if *var == tyvar => {
            *in_typ = to_typ.clone();
        }
        Type::FuncType(func) => {
            subst_var_typ(tyvar.clone(), to_typ.clone(), &mut func.lhs);
            subst_var_typ(tyvar, to_typ, &mut func.rhs);
        }
        // Add cases for other Type variants if needed
        _ => {}
    }
}

// Checks if a type variable occurs in a type
fn occurs(tyvar: &TypeVar, typ: &Type) -> bool {
    match typ {
        Type::TypeVar(other_tyvar) => tyvar == other_tyvar,
        Type::FuncType(func_type) => occurs(tyvar, &func_type.lhs) || occurs(tyvar, &func_type.rhs),
        _ => false,
    }
}

// extends curr with other, in place
fn compose(curr: &mut Substitution, other: Substitution) {
    curr.extend(other);
}

fn compose_constraint_info(curr: &mut ConstraintInfo, other: Option<ConstraintInfo>) {
    if other.is_some() {
        let other = other.unwrap();
        curr.dimexprs.extend(other.dimexprs);
        curr.dimvars.extend(other.dimvars);
    }
}

// NOTE: Add a type scheme, with a ForAll quantifier (see article)
