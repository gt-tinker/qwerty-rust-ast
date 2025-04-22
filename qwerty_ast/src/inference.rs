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
// pub fn infer(node: &mut Program) {
//     let mut tyvar_generator = TypeVarGenerator::new();
//     let mut root_ctx = instantiate_global_map();
//     for child in &mut node.body {
//         infer_helper(child, &mut tyvar_generator, &mut root_ctx);
//     }
//
//     // dbg!(root_ctx);
// }

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
        _ => (Substitution::new(), Some(ConstraintInfo::default())),
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
