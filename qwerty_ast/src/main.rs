#![allow(unused_imports, dead_code)]
/// This module serves as a testbed and design area for the new Qwerty AST, built in Rust
/// The goal of this particular file is to test implementations that make the AST the
/// most ergonomic to use!
mod ast;
mod basis;
mod dimexpr;
mod inference;
mod types;

fn main() {
    // TODO: Add tests that create ASTs based on what the
    // frontend *would* parse, then do testing for typechecking later
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast::ASTNode;
    use dimexpr::DimExpr;
    use dimexpr::*;
    use inference::{infer, ConstraintInfo, Context, DimConstraint};
    use std::collections::HashMap;
    use std::str::FromStr;
    use types::{
        Eigenbit, FuncType, Order, PrimitiveBasis, RegisterType, RegisterVariant, Type, UnitType,
    };

    macro_rules! ast_node {
        ($id:ident, $( $field:ident : $val:expr ),*) => {
            Box::new(ASTNode::$id(ast::$id {
                dbg: ast::DebugInfo::default(),
                $( $field : $val, )*
            }))
        };
    }

    macro_rules! ast_node_struct {
        ($id:ident, $( $field:ident : $val:expr ),*) => {
            ast::$id {
                dbg: ast::DebugInfo::default(),
                $( $field : $val, )*
            }
        };
    }

    macro_rules! ast_dim {
        ($dim_str:expr) => {
            dimexpr::DimExpr::from_str($dim_str).unwrap()
        };
    }

    // Defining the global context (temporary instance for now, can be done
    // in a separate file)
    // Context should be for all user-defined terms, such as functions and variables
    fn instantiate_global_map() -> Context {
        let mut ctx: HashMap<String, Type> = HashMap::new();

        // NOTE: We should add defined functions such as the built-in bases,
        // flip, id, sign, etc., later

        ctx
    }

    #[test]
    fn test_simple_ret_infer() {
        let program = r#"
        @qpu
        def kernel():
            return 'p'
        "#;
        println!("Simple test: {}", program);

        let prog = |ret_val| {
            ast_node_struct! {Program,
                ctx: instantiate_global_map(),
                body: vec![
                    *ast_node!{Function,
                        typ: None,
                        name: "kernel".to_string(),
                        args: vec![],
                        constraint_info: ConstraintInfo::default(),
                        is_classical: false,
                        body: vec![
                            *ast_node!{Return,
                                typ: None,
                                value: ret_val
                            }
                        ]
                    }
                ]
            }
        };

        let ret_val = ast_node! {QubitLiteral,
            typ: None,
            factor: ast_dim!["1"],
            elts: vec![
                ast_node_struct!{QubitSymbol,
                    basis: PrimitiveBasis::Pm,
                    eigenbit: Eigenbit::One
                },
            ]
        };

        let mut ast = prog(ret_val);

        infer(&mut ast);

        let func_type = Type::FuncType(FuncType {
            lhs: Box::new(Type::UnitType()),
            rhs: Box::new(Type::RegisterType(RegisterType {
                variant: RegisterVariant::Qubit,
                dim: ast_dim!["1"],
            })),
            is_rev: false,
        });

        let ker_type = ast.body[0].get_typ().unwrap();

        assert_eq!(ker_type, func_type);
    }

    #[test]
    fn test_pipe_infer() {
        let program = r#"
        @qpu[[N]]
        def kernel():
            return 'p'[N] | std.flip[5]
        "#;
        println!("Pipe & first dimvar test: {}", program);

        let prog = |ret_val| {
            ast_node_struct! {Program,
                ctx: instantiate_global_map(),
                body: vec![
                    *ast_node!{Function,
                        typ: None, // Should be Unit -> Qubit[5]
                        name: "kernel".to_string(),
                        args: vec![],
                        constraint_info: ConstraintInfo::default(),
                        is_classical: false,
                        body: vec![
                            *ast_node!{Return,
                                typ: None, // Qubit[5]
                                value: ret_val
                            }
                        ]
                    }
                ]
            }
        };

        // NOTE: This *could* feasibly be wrapped in a tensor node, where the tensor has the N
        // dim-var and the QubitLiteral has the single QubitSymbol with dim 1 (or or we get rid of
        // QubitLiteral as a concept, even though it technically works with the type system better)

        let ret_val = ast_node! {Pipe,
            typ: None,
            lhs: ast_node!{QubitLiteral,
                    typ: None,
                    factor: ast_dim!["N"],
                    elts: vec![
                        ast_node_struct!{QubitSymbol,
                            basis: PrimitiveBasis::Pm,
                            eigenbit: Eigenbit::Zero
                        },
                    ]
                },
            rhs: ast_node!{Flip,
                typ: None,
                basis: ast_node!{Basis,
                    typ: None,
                    dim: ast_dim!["5"], // Unsure if this should be 1 or 5
                    elts: vec![
                        basis::BasisElement::BuiltInBasis(
                            ast_node_struct!{BuiltInBasis,
                                typ: None,
                                prim_basis: PrimitiveBasis::Std,
                                dim: ast_dim!["5"]
                            }
                        )
                    ]
                }
            }
        };

        let mut ast = prog(ret_val);

        infer(&mut ast);

        // dbg!(&ast);

        let func_type = Type::FuncType(FuncType {
            lhs: Box::new(Type::UnitType()),
            rhs: Box::new(Type::RegisterType(RegisterType {
                variant: RegisterVariant::Qubit,
                dim: ast_dim!["5"],
            })),
            is_rev: false,
        });

        let ker_type = ast.body[0].get_typ().unwrap();

        assert_eq!(ker_type, func_type);
    }

    #[test]
    fn test_bell() {
        let program = r#"
        @qpu[[N]]
        def kernel() -> bit[2]:
             return 'p0' | '1' & std.flip | std[2].measure
        "#;
        println!("Bell test: {}", program);

        let prog = |pipeline_expr| {
            ast_node_struct! {Program,
                ctx: instantiate_global_map(),
                body: vec![
                    *ast_node!{Function,
                        typ: None,
                        name: "kernel".to_string(),
                        args: vec![],
                        constraint_info: ConstraintInfo::default(),
                        is_classical: false,
                        body: vec![
                            *ast_node!{Return,
                                typ: None,
                                value: pipeline_expr
                            }
                        ]
                    }
                ]
            }
        };

        let pipeline_expr = ast_node! {Pipe,
            typ: None,
            lhs: ast_node!{Pipe,
                typ: None,
                lhs: ast_node!{QubitLiteral,
                    typ: None,
                    factor: ast_dim!["1"],
                    elts: vec![
                        ast_node_struct!{QubitSymbol,
                            basis: PrimitiveBasis::Pm,
                            eigenbit: Eigenbit::Zero
                        },
                        ast_node_struct!{QubitSymbol,
                            basis: PrimitiveBasis::Std,
                            eigenbit: Eigenbit::Zero
                        },
                    ]
                },
                rhs: ast_node!{Pred,
                    typ: None,
                    order: Order::Unknown,
                    basis: ast_node!{QubitLiteral,
                        typ: None,
                        factor: ast_dim!["1"],
                        elts: vec![
                            ast_node_struct!{QubitSymbol,
                                basis: PrimitiveBasis::Std,
                                eigenbit: Eigenbit::One
                            },
                        ]
                    },
                    body: ast_node!{Flip,
                        typ: None,
                        basis: ast_node!{Basis,
                            typ: None,
                            dim: ast_dim!["1"],
                            elts: vec![
                                basis::BasisElement::BuiltInBasis(
                                    ast_node_struct!{BuiltInBasis,
                                        typ: None,
                                        prim_basis: PrimitiveBasis::Std,
                                        dim: ast_dim!["1"]
                                    }
                                )
                            ]
                        }
                    }
                }
            },
            rhs: ast_node!{Measure,
                typ: None,
                basis: ast_node!{Basis,
                    typ: None,
                    dim: ast_dim!["2"],
                    elts: vec![
                        basis::BasisElement::BuiltInBasis(
                            ast_node_struct!{BuiltInBasis,
                                typ: None,
                                prim_basis: PrimitiveBasis::Std,
                                dim: ast_dim!["2"]
                            }
                        )
                    ]
                }
            }
        };

        let mut ast = prog(pipeline_expr);

        infer(&mut ast);

        // dbg!(&ast);

        let func_type = Type::FuncType(FuncType {
            lhs: Box::new(Type::UnitType()),
            rhs: Box::new(Type::RegisterType(RegisterType {
                variant: RegisterVariant::Bit,
                dim: ast_dim!["2"],
            })),
            is_rev: false,
        });

        let ker_type = ast.body[0].get_typ().unwrap();

        assert_eq!(ker_type, func_type);
    }

    #[test]
    fn test_dimvar_infer() {
        let program = r#"
        @qpu[[N]]
        def kernel():
            return 'p'[N+1] | id[M+2] | std.flip[5]
        "#;
        println!("Pipe & second dimvar test: {}", program);

        let prog = |ret_val| {
            ast_node_struct! {Program,
                ctx: instantiate_global_map(),
                body: vec![
                    *ast_node!{Function,
                        typ: None, // Should be Unit -> Qubit[5]
                        name: "kernel".to_string(),
                        args: vec![],
                        constraint_info: ConstraintInfo::default(),
                        is_classical: false,
                        body: vec![
                            *ast_node!{Return,
                                typ: None, // Qubit[5]
                                value: ret_val
                            }
                        ]
                    }
                ]
            }
        };

        let ret_val = ast_node! {Pipe,
            typ: None,
            lhs: ast_node! {Pipe,
                typ: None,
                lhs: ast_node!{QubitLiteral,
                    typ: None,
                    factor: ast_dim!["N+1"],
                    elts: vec![
                        ast_node_struct!{QubitSymbol,
                            basis: PrimitiveBasis::Pm,
                            eigenbit: Eigenbit::Zero
                        },
                    ]
                },
                rhs: ast_node! {Identity,
                    typ: None,
                    dim: ast_dim!["M+2"]
                }
            },
            rhs: ast_node!{Flip,
                typ: None,
                basis: ast_node!{Basis,
                    typ: None,
                    dim: ast_dim!["5"], // Unsure if this should be 1 or 5
                    elts: vec![
                        basis::BasisElement::BuiltInBasis(
                            ast_node_struct!{BuiltInBasis,
                                typ: None,
                                prim_basis: PrimitiveBasis::Std,
                                dim: ast_dim!["5"]
                            }
                        )
                    ]
                }
            }
        };

        let mut ast = prog(ret_val);

        infer(&mut ast);

        // dbg!(&ast);

        let func_type = Type::FuncType(FuncType {
            lhs: Box::new(Type::UnitType()),
            rhs: Box::new(Type::RegisterType(RegisterType {
                variant: RegisterVariant::Qubit,
                dim: ast_dim!["5"],
            })),
            is_rev: false,
        });

        let ker_type = ast.body[0].get_typ().unwrap();

        assert_eq!(ker_type, func_type);
    }

    // #[test]
    // fn test_teleport() {
    //     // The goal of this test is to see if we can make
    //     // teleportation work, by perhaps having an input with a defined type
    //     // and then using HM to reconstruct the types with it
    //     // (Try maintaining [N] DimExpr in AST, then convert)
    //
    //     let program = r#"
    //     @qpu
    //     def teleport(secret: qubit) -> qubit:
    //         alice, bob = 'p0' | '1' & std.flip
    //
    //         m_pm, m_std = secret + alice | '1' & std.flip \
    //                                      | (pm + std).measure
    //
    //         secret_teleported = \
    //             bob | (pm.flip if m_std else id) \
    //                 | (std.flip if m_pm else id)
    //
    //         return secret_teleported
    //
    //     @qpu(teleport)
    //     def kernel(teleport: qfunc) -> bit:
    //         # Try changing this to a different state
    //         example = 'j'
    //         return teleport(example) | ij.measure
    //     "#;
    //     println!("Teleport test: {}", program);
    //
    //     let p0 = basis::QubitLiteral {
    //         elts: vec![
    //             basis::QubitSymbol {
    //                 basis: types::PrimitiveBasis::Pm,
    //                 eigenbit: types::Eigenbit::Zero,
    //                 dbg: ast::DebugInfo::default(),
    //             },
    //             basis::QubitSymbol {
    //                 basis: types::PrimitiveBasis::Std,
    //                 eigenbit: types::Eigenbit::Zero,
    //                 dbg: ast::DebugInfo::default(),
    //             },
    //         ],
    //         factor: ast_dim!["2"],
    //         typ: None, // Should be RegisterType { variant: Qubit, dim: 2 }
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     let pred_basis = basis::BasisLiteral {
    //         literal: basis::QubitLiteral {
    //             elts: vec![basis::QubitSymbol {
    //                 basis: types::PrimitiveBasis::Std,
    //                 eigenbit: types::Eigenbit::One,
    //                 dbg: ast::DebugInfo::default(),
    //             }],
    //             factor: ast_dim!["1"],
    //             typ: None, // Should be RegisterType { variant: Qubit, dim: 1}
    //             dbg: ast::DebugInfo::default(),
    //         },
    //         phase: None,
    //         typ: None, // Should be BasisType { dim: 1}
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     let flip_std_basis = basis::BuiltInBasis {
    //         prim_basis: types::PrimitiveBasis::Std,
    //         dim: ast_dim!["1"],
    //         typ: None, // Should be BasisType { dim: 1 }
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     let flip_std = ast::Flip {
    //         basis: Box::new(ASTNode::BuiltInBasis(flip_std_basis)),
    //         typ: None, // Should be (Qubit[1] -rev-> Qubit[1])
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     let pred = ast::Pred {
    //         order: types::Order::BU, // What is this field for?
    //         basis: Box::new(ASTNode::BasisLiteral(pred_basis)),
    //         body: Box::new(ASTNode::Flip(flip_std.clone())),
    //         typ: None, // Should be (Qubit[2] -rev-> Qubit[2])
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     let pipe1 = ast::Pipe {
    //         lhs: Box::new(ASTNode::QubitLiteral(p0)),
    //         rhs: Box::new(ASTNode::Pred(pred.clone())),
    //         typ: None, // Should be Qubit[2]
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // Create assignment
    //     let assignment1 = ast::Assign {
    //         assigned_to: vec!["alice".to_string(), "bob".to_string()],
    //         val: Box::new(ASTNode::Pipe(pipe1)),
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // Create basis for secret + alice
    //     let secret_plus_alice = ast::Tensor {
    //         elts: vec![
    //             ASTNode::Variable("secret".to_string()),
    //             ASTNode::Variable("alice".to_string()),
    //         ],
    //         factor: ast_dim!["2"],
    //         typ: None, // RegisterType (Qubit[2]) ?
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // '1' & std.flip (rename, cloned earlier)
    //     let pred2 = pred;
    //
    //     let pipe2 = ast::Pipe {
    //         lhs: Box::new(ASTNode::Tensor(secret_plus_alice)),
    //         rhs: Box::new(ASTNode::Pred(pred2)),
    //         typ: None, // Should be Qubit[2]
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // (pm + std).measure
    //     let measure_basis2 = basis::Basis {
    //         elts: vec![
    //             basis::BasisElement::BuiltInBasis(basis::BuiltInBasis {
    //                 prim_basis: types::PrimitiveBasis::Pm,
    //                 dim: ast_dim!["1"],
    //                 typ: None, // Should be BasisType { dim: 1 }
    //                 dbg: ast::DebugInfo::default(),
    //             }),
    //             basis::BasisElement::BuiltInBasis(basis::BuiltInBasis {
    //                 prim_basis: types::PrimitiveBasis::Std,
    //                 dim: ast_dim!["1"],
    //                 typ: None, // Should be BasisType { dim: 1 }
    //                 dbg: ast::DebugInfo::default(),
    //             }),
    //         ],
    //         dim: ast_dim!["2"],
    //         typ: None, // BasisType (Qubit[2]) ?
    //         // NOTE: To self, what is BasisType supposed to be
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     let pm_std_measure = ast::Measure {
    //         basis: Box::new(ASTNode::Basis(measure_basis2)),
    //         typ: None, // (Qubit[2] -> Bit[2])
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // pipe3 | measure
    //     let pipe3 = ast::Pipe {
    //         lhs: Box::new(ASTNode::Pipe(pipe2)),
    //         rhs: Box::new(ASTNode::Measure(pm_std_measure)),
    //         typ: None, // Should be Bit[2]
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // m_pm, m_std assignment
    //     let assignment2 = ast::Assign {
    //         assigned_to: vec!["m_pm".to_string(), "m_std".to_string()],
    //         val: Box::new(ASTNode::Pipe(pipe3)),
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     let flip_pm_basis = basis::BuiltInBasis {
    //         prim_basis: types::PrimitiveBasis::Pm,
    //         dim: ast_dim!["1"],
    //         typ: None, // Should be BasisType { dim: 1 }
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     let flip_pm = ast::Flip {
    //         basis: Box::new(ASTNode::BuiltInBasis(flip_pm_basis)),
    //         typ: None, // Should be (Qubit[1] -rev-> Qubit[1])
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     let identity = ast::Identity {
    //         typ: None, // Should be (Qubit[1] -rev-> Qubit[1])
    //         dim: ast_dim!["1"],
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // cond1 = set up conditional (pm.flip if m_std else id)
    //     let cond1 = ast::Conditional {
    //         if_expr: Box::new(ASTNode::Variable("m_std".to_string())),
    //         then_expr: Box::new(ASTNode::Flip(flip_pm)),
    //         else_expr: Box::new(ASTNode::Identity(identity.clone())),
    //         typ: None, // Should be (Qubit[1] -rev-> Qubit[1])
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // pipe4 = bob | cond1
    //     let pipe4 = ast::Pipe {
    //         // Can we use variables a different way?
    //         lhs: Box::new(ASTNode::Variable("bob".to_string())),
    //         rhs: Box::new(ASTNode::Conditional(cond1)),
    //         typ: None, // Should be Qubit[1]
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // cond2 = set up conditional (std.flip if m_std else id)
    //     let cond2 = ast::Conditional {
    //         if_expr: Box::new(ASTNode::Variable("m_pm".to_string())),
    //         then_expr: Box::new(ASTNode::Flip(flip_std)),
    //         else_expr: Box::new(ASTNode::Identity(identity)),
    //         typ: None, // Should be (Qubit[1] -rev-> Qubit[1])
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // pipe5 = pipe4 | cond2
    //     let pipe5 = ast::Pipe {
    //         // Can we use variables a different way?
    //         lhs: Box::new(ASTNode::Pipe(pipe4)),
    //         rhs: Box::new(ASTNode::Conditional(cond2)),
    //         typ: None, // Should be Qubit[1]
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // secret_teleported = pipe5
    //     let assignment3 = ast::Assign {
    //         assigned_to: vec!["secret_teleported".to_string()],
    //         val: Box::new(ASTNode::Pipe(pipe5)),
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // return secret_teleported
    //     let ret_teleport = ast::Return {
    //         // FIXME: Should we return a Variable here that corresponds
    //         // to the value given at assignment? Or the assignment node itself?
    //         // value: Box::new(ASTNode::Assign(assignment3)),
    //         value: Box::new(ASTNode::Variable("secret_teleported".to_string())),
    //         typ: None, // Should be Qubit[1]
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // Teleport function //
    //
    //     let teleport = ast::Function {
    //         name: "teleport".to_string(),
    //         args: vec![ASTNode::Variable("secret".to_string())],
    //         body: vec![
    //             ASTNode::Assign(assignment1),
    //             ASTNode::Assign(assignment2),
    //             ASTNode::Assign(assignment3),
    //             ASTNode::Return(ret_teleport),
    //         ],
    //         constraint_info: ConstraintInfo::default(),
    //         is_classical: false,
    //         typ: None, // Should be FuncType (Qubit -> Qubit)
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // Kernel Function //
    //
    //     let example = basis::QubitSymbol {
    //         basis: types::PrimitiveBasis::Ij,
    //         eigenbit: types::Eigenbit::One,
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // Call
    //     let call_teleport = ast::Call {
    //         name: "teleport".to_string(),
    //         ins: vec![ASTNode::QubitLiteral(ast::QubitLiteral {
    //             elts: vec![example.clone()],
    //             factor: ast_dim!["1"],
    //             typ: None, // Qubit[1]
    //             dbg: ast::DebugInfo::default(),
    //         })],
    //         typ: None, // Qubit[1] -> Qubit[1]
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // Measure
    //     let measure_basis_ij = basis::BuiltInBasis {
    //         prim_basis: types::PrimitiveBasis::Ij,
    //         dim: ast_dim!["1"],
    //         typ: None, // Should be BasisType { dim: 1 }
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     let measure_ij = ast::Measure {
    //         basis: Box::new(ASTNode::BuiltInBasis(measure_basis_ij)),
    //         typ: None, // Should be (Qubit[1] -> Bit[1])
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // Pipe
    //     let pipe6 = ast::Pipe {
    //         // Can we use variables a different way?
    //         lhs: Box::new(ASTNode::Call(call_teleport)),
    //         rhs: Box::new(ASTNode::Measure(measure_ij)),
    //         typ: None, // Should be Bit[1]
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // Return statement
    //     let ret_kernel = ast::Return {
    //         value: Box::new(ASTNode::Pipe(pipe6)),
    //         typ: None, // Should be Bit[1]
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // kernel function
    //     let kernel = ast::Function {
    //         name: "kernel".to_string(),
    //         args: vec![ASTNode::Function(teleport.clone())],
    //         body: vec![
    //             ASTNode::QubitLiteral(ast::QubitLiteral {
    //                 elts: vec![example],
    //                 factor: ast_dim!["1"],
    //                 typ: None, // Qubit[1]
    //                 dbg: ast::DebugInfo::default(),
    //             }),
    //             ASTNode::Return(ret_kernel),
    //         ],
    //         constraint_info: ConstraintInfo::default(),
    //         is_classical: false,
    //         typ: None, // FuncType ((Qubit -> Qubit) -> Bit)
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     // NOTE: To self: Create Program node at the end
    //     let mut program = ast::Program {
    //         body: vec![ASTNode::Function(teleport), ASTNode::Function(kernel)],
    //         // FIXME: Fix the ctx later!
    //         ctx: instantiate_global_map(),
    //         dbg: ast::DebugInfo::default(),
    //     };
    //
    //     infer(&mut program);
    //
    //     dbg!(program);
    //
    //     assert!(true);
    // } // end test
}
