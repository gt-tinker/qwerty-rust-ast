/// Contains code for Dimension DimExprs (as trees)
use crate::inference::{DimVarInfo, SimplifiedConstraint};
use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Sub};

// NOTE: Goal is to create a tree and a solver for systems of equations for DimExprs

// from defs.hpp
pub type DimVar = String;
pub type DimVarValue = isize;

#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Exp,
}

// Kinda redundant now, but good for splitting functionality clearly
#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub enum UnaryOp {
    USub,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum DimExpr {
    Binary(BinOp, Box<DimExpr>, Box<DimExpr>),
    Unary(UnaryOp, Box<DimExpr>),
    Number(DimVarValue),
    Variable(String),
}

impl DimExpr {
    fn eval(&mut self, var_vals: &HashMap<DimVar, DimVarValue>) -> DimVarValue {
        match self {
            DimExpr::Number(n) => *n,
            DimExpr::Unary(_negative, expr) => -1 * expr.eval(var_vals),
            DimExpr::Binary(BinOp::Add, expr1, expr2) => {
                expr1.eval(var_vals) + expr2.eval(var_vals)
            }
            DimExpr::Binary(BinOp::Mul, expr1, expr2) => {
                expr1.eval(var_vals) * expr2.eval(var_vals)
            }
            DimExpr::Binary(BinOp::Sub, expr1, expr2) => {
                expr1.eval(var_vals) - expr2.eval(var_vals)
            }
            DimExpr::Binary(BinOp::Exp, expr1, expr2) => {
                expr1.eval(var_vals).pow(expr2.eval(var_vals) as u32)
            }
            DimExpr::Binary(BinOp::Div, expr1, expr2) => {
                expr1.eval(var_vals) / expr2.eval(var_vals)
            }
            DimExpr::Variable(var) => *var_vals.get(var).unwrap(),
        }
    }

    // Traverse DimExpr tree, see if there are variables
    pub fn is_constant(&self) -> bool {
        fn recursive_checker(node: &DimExpr) -> bool {
            match node {
                DimExpr::Number(_) => true,
                DimExpr::Binary(_, expr1, expr2) => {
                    recursive_checker(expr1) & recursive_checker(expr2)
                }
                DimExpr::Unary(_, expr) => recursive_checker(expr),
                DimExpr::Variable(_) => return false,
            }
        }
        recursive_checker(&self)
    }

    // FIXME: We can make a more powerful version of this with
    // all the passes included!
    pub fn simplify(&mut self) {
        match self {
            DimExpr::Number(_) | DimExpr::Variable(_) => {}
            DimExpr::Unary(_negative, expr) => {
                expr.simplify();
                if let DimExpr::Number(val) = **expr {
                    *self = DimExpr::Number(-val);
                }
            }
            DimExpr::Binary(op, left, right) => {
                // Simplify sides, then join
                left.simplify();
                right.simplify();

                // Check if both sides are numbers, then operate!
                if let (DimExpr::Number(l), DimExpr::Number(r)) = (&**left, &**right) {
                    let result = match op {
                        BinOp::Add => l + r,
                        BinOp::Sub => l - r,
                        BinOp::Mul => l * r,
                        BinOp::Div => {
                            if *r == 0 {
                                panic!("Divide by 0 error");
                            }
                            l / r
                        }
                        BinOp::Exp => l.pow(*r as u32),
                    };

                    *self = DimExpr::Number(result);
                    // Cases below are for 1 * N or N * 1
                } else if let BinOp::Mul = op {
                    if let (DimExpr::Number(1), DimExpr::Variable(var)) = (&**left, &**right) {
                        *self = DimExpr::Variable(var.to_string());
                    } else if let (DimExpr::Variable(var), DimExpr::Number(1)) = (&**left, &**right)
                    {
                        *self = DimExpr::Variable(var.to_string());
                    } else if let (_, DimExpr::Number(0)) = (&**left, &**right) {
                        // For thing * 0 or 0 * thing
                        *self = DimExpr::Number(0);
                    } else if let (DimExpr::Number(0), _) = (&**left, &**right) {
                        *self = DimExpr::Number(0);
                    }
                } else if let BinOp::Add = op {
                    // Simplify thing + 0 or 0 + thing to thing
                    if let (_, DimExpr::Number(0)) = (&**left, &**right) {
                        // For thing * 0 or 0 * thing
                        *self = *left.clone(); // TODO: Can we get rid of this?
                    } else if let (DimExpr::Number(0), _) = (&**left, &**right) {
                        *self = *right.clone(); // TODO: Can we get rid of this?
                    }
                }
            }
        }
    }

    // A canonicalization pass we need, for simplicity
    pub fn transform_sub_to_add(&mut self) {
        match self {
            DimExpr::Binary(BinOp::Sub, left, right) => {
                // Transform A - B into A + (-1 * B)
                // NOTE: We update left as well, just in case
                left.transform_sub_to_add();
                right.transform_to_negate_with_mul();
                right.transform_sub_to_add();
                *self = DimExpr::Binary(BinOp::Add, left.take(), right.take());
            }
            DimExpr::Binary(_, left, right) => {
                // Recursively apply transformation to children
                left.transform_sub_to_add();
                right.transform_sub_to_add();
            }
            DimExpr::Unary(_, expr) => {
                // Apply transformation to child expression
                expr.transform_sub_to_add();
            }
            DimExpr::Number(_) | DimExpr::Variable(_) => {
                // No transformation needed
            }
        }
        self.simplify();
    }

    fn transform_to_negate_with_mul(&mut self) {
        // Replace the expression with -1 * original_expr
        *self = DimExpr::Binary(BinOp::Mul, Box::new(DimExpr::Number(-1)), self.take());
    }

    // We need a manual implementation of take because
    // deriving or implementing default is annoying to
    // work with, even if it's simple
    fn take(&mut self) -> Box<DimExpr> {
        // Replace the current expression with a placeholder and return the old one
        Box::new(std::mem::replace(self, DimExpr::Number(0)))
    }

    // utility function to swap lhs and rhs of BinOp
    fn swap_binop(&mut self) {
        if let DimExpr::Binary(_, left, right) = self {
            std::mem::swap(left, right);
        }
    }

    // Tries to apply associativity rules; returns true if
    // a change was made
    fn apply_assoc_helper(&mut self) -> bool {
        if let DimExpr::Binary(op, left, right) = self {
            // Recursively apply associativity to both children
            left.apply_assoc();
            right.apply_assoc();

            // If both sides are binary operations with the same operator, flatten constants
            if let (DimExpr::Binary(inner_op, inner_left, inner_right), DimExpr::Number(_)) =
                (&mut **left, &**right)
            {
                // NOTE: We cover all cases with swap_binop
                if inner_op == op {
                    if let DimExpr::Number(_) = **inner_right {
                        // Group constants together: (N + 2) + 1 => N + (2 + 1)
                        // Instead of just adding the value, change the node pointers
                        std::mem::swap(inner_left, right);
                        left.simplify();
                        return true;
                    } else if let DimExpr::Number(_) = **inner_left {
                        std::mem::swap(inner_right, right);
                        left.simplify();
                        return true;
                    }
                }
            } else if let (DimExpr::Number(_), DimExpr::Binary(inner_op, inner_left, inner_right)) =
                (&**left, &mut **right)
            {
                if inner_op == op {
                    if let DimExpr::Number(_) = **inner_left {
                        // Group constants together: 1 + (N + 2) => (1 + 2) + N
                        std::mem::swap(inner_right, left);
                        right.simplify();
                        return true;
                    } else if let DimExpr::Number(_) = **inner_left {
                        std::mem::swap(inner_right, left);
                        left.simplify();
                        return true;
                    }
                }
            }
        }
        false
    }

    // Applies associative rule with swaps, using helper
    fn apply_assoc_with_swap(&mut self) -> bool {
        if self.apply_assoc_helper() {
            return true;
        }

        self.swap_binop();
        if self.apply_assoc_helper() {
            return true;
        }

        self.swap_binop();
        false
    }

    pub fn apply_assoc(&mut self) {
        while self.apply_assoc_with_swap() {
            self.simplify();
        }
        self.simplify();
    }

    // Applies distributive rule
    fn apply_distrib(&mut self) -> bool {
        if let DimExpr::Binary(op, left, right) = self {
            // Recursively apply distributivity to both children
            left.apply_distrib();
            right.apply_distrib();

            let cont = match op {
                BinOp::Mul | BinOp::Div => true,
                _ => false,
            };

            if cont {
                if let (DimExpr::Binary(_, inner_left, inner_right), DimExpr::Number(_)) =
                    (&mut **left, &**right)
                {
                    // NOTE:
                    // left = binop, right = num
                    // inner_left, inner right of left

                    // NOTE: Steps
                    // 1. Create DimExpr with matching op using (inner_left, right), and
                    //    (inner_right, right)
                    let new_left = DimExpr::Binary(op.clone(), inner_left.take(), right.clone());
                    let new_right = DimExpr::Binary(op.clone(), inner_right.take(), right.clone());
                    // 2. Change self's op to Add
                    *op = BinOp::Add;
                    // 3. Replace left with (inner_left, right)
                    let _ = std::mem::replace(left, Box::new(new_left));
                    // 4. Replace right with (inner_right, right)
                    let _ = std::mem::replace(right, Box::new(new_right));
                    // 5. Done!
                    self.simplify();
                    return true;
                } else if let (DimExpr::Number(_), DimExpr::Binary(_, inner_left, inner_right)) =
                    (&**left, &mut **right)
                {
                    // NOTE:
                    // left = num, right = binop
                    // inner_left, inner right of right

                    // NOTE: Steps
                    // 1. Create DimExpr with matching op using (inner_left, left), and
                    //    (inner_right, left)
                    let new_left = DimExpr::Binary(op.clone(), inner_left.take(), left.clone());
                    let new_right = DimExpr::Binary(op.clone(), inner_right.take(), left.clone());
                    // 2. Change self's op to Add
                    *op = BinOp::Add;
                    // 3. Replace left with (inner_left, right)
                    let _ = std::mem::replace(left, Box::new(new_left));
                    // 4. Replace right with (inner_right, right)
                    let _ = std::mem::replace(right, Box::new(new_right));
                    // 5. Done!
                    self.simplify();
                    return true;
                }
            }
        }
        false
    }

    pub fn simplify_with_passes(&mut self) {
        self.apply_distrib();
        self.transform_sub_to_add();
        self.apply_distrib();
        self.apply_assoc();
        self.simplify();
    }

    pub fn symmetry(&self, other: &Self) -> SimplifiedConstraint {
        let lhs_constraint = self.generate_simplified_constraint();
        let rhs_constraint = other.generate_simplified_constraint();
        let combined_constraint =
            SimplifiedConstraint::normalize_constraints(&lhs_constraint, &rhs_constraint);
        combined_constraint
    }

    fn generate_simplified_constraint(&self) -> SimplifiedConstraint {
        let mut constraints = SimplifiedConstraint::default();
        self.simplified_constraint_helper(&mut constraints);
        constraints
    }

    fn simplified_constraint_helper(&self, constraints: &mut SimplifiedConstraint) {
        // If you see a constant, add it to the `sum_consts` field
        // if you see a binop with * or /, process it as a DimVarInfo
        // and add it to `vars`
        match self {
            DimExpr::Binary(op, left, right) => {
                match &op {
                    BinOp::Add | BinOp::Sub => {
                        // handle num case
                        left.simplified_constraint_helper(constraints);
                        right.simplified_constraint_helper(constraints);
                    }
                    BinOp::Mul | BinOp::Div => {
                        // handle coeff case
                        match (&**left, &**right) {
                            (DimExpr::Number(coeff), DimExpr::Variable(dimvar))
                            | (DimExpr::Variable(dimvar), DimExpr::Number(coeff)) => {
                                constraints.vars.push(DimVarInfo {
                                    name: dimvar.to_string(),
                                    op: op.clone(),
                                    coeff: *coeff,
                                });
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            DimExpr::Number(val) => constraints.sum_consts += val,
            DimExpr::Variable(name) => constraints.vars.push(DimVarInfo {
                name: name.to_string(),
                op: BinOp::Mul,
                coeff: 1,
            }),
            _ => {}
        }
    }
}

#[test]
fn test_simplified_const() {
    println!("simplified constraint test");
    let e7: DimExpr = "1 + N + M + 2 + 3".parse().unwrap();
    let e8: DimExpr = "1 + 2*M + 3*N + 4".parse().unwrap();
    dbg!(&e7);
    let lhs = e7.generate_simplified_constraint();
    dbg!(&lhs);
    let rhs = e8.generate_simplified_constraint();
    dbg!(&rhs);

    let res = SimplifiedConstraint::normalize_constraints(&lhs, &rhs);
    dbg!(res);
}

// NOTE: More for visual inspection than a rigorous test
#[test]
fn test_associativity() {
    dbg!("associativity test");
    let mut e1: DimExpr = "(N+2)+1".parse().unwrap();
    let mut e2: DimExpr = "1+(N+2)".parse().unwrap();
    let mut e3: DimExpr = "5+1+(N+2)+3+4".parse().unwrap();
    let mut e4: DimExpr = "(N-2)-1".parse().unwrap();
    let mut e5: DimExpr = "1-(N-2)".parse().unwrap();
    let mut e6: DimExpr = "5-1-(N-2)-3".parse().unwrap();
    println!("---------------PRIOR----------------");
    e1.apply_assoc();
    dbg!(e1);
    println!("---------------PRIOR----------------");
    e2.apply_assoc();
    dbg!(e2);
    println!("---------------PRIOR----------------");
    e3.apply_assoc();
    dbg!(e3);
    println!("---------------PRIOR----------------");
    dbg!(&e4);
    e4.apply_distrib();
    e4.transform_sub_to_add();
    e4.apply_assoc();
    dbg!(e4);
    println!("---------------PRIOR----------------");
    e5.transform_sub_to_add();
    e5.apply_distrib();
    e5.apply_assoc();
    dbg!(e5);
    println!("---------------PRIOR----------------");
    dbg!(&e6);
    e6.transform_sub_to_add();
    dbg!(&e6);
    e6.apply_distrib();
    e6.transform_sub_to_add();
    dbg!(&e6);
    e6.apply_assoc();
    dbg!(e6);
}

#[test]
fn test_distributivity() {
    dbg!("distributivity test");
    let mut e1: DimExpr = "(N+2)*3".parse().unwrap();
    let mut e2: DimExpr = "3*(N+2)".parse().unwrap();
    let mut e3: DimExpr = "(N+2)/3".parse().unwrap();
    let mut e4: DimExpr = "3/(N+2)".parse().unwrap();
    let mut e5: DimExpr = "3*(N*2)".parse().unwrap();
    println!("---------------PRIOR----------------");
    e1.apply_distrib();
    dbg!(e1);
    println!("---------------PRIOR----------------");
    e2.apply_distrib();
    dbg!(e2);
    println!("---------------PRIOR----------------");
    e3.apply_distrib();
    dbg!(e3);
    println!("---------------PRIOR----------------");
    e4.apply_distrib();
    dbg!(e4);
    println!("---------------PRIOR----------------");
    e5.apply_distrib();
    dbg!(e5);
}

#[test]
fn test_overall_simplify() {
    let mut e1: DimExpr = "3*(N+2)".parse().unwrap();
    let mut e2: DimExpr = "(N+2)/3".parse().unwrap();
    let mut e3: DimExpr = "(N-2)-1".parse().unwrap();
    let mut e4: DimExpr = "5-1-(N-2)-3".parse().unwrap();
    println!("---------------PRIOR----------------");
    e1.simplify_with_passes();
    dbg!(e1);
    println!("---------------PRIOR----------------");
    e2.simplify_with_passes();
    dbg!(e2);
    println!("---------------PRIOR----------------");
    e3.simplify_with_passes();
    dbg!(e3);
    println!("---------------PRIOR----------------");
    e4.simplify_with_passes();
    dbg!(e4);
}

impl std::str::FromStr for DimExpr {
    type Err = String;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let tokens = tokenize(input)?;
        let mut parser = Parser::new(tokens);
        parser.parse_expr()
    }
}

// For generating constant DimExprs
impl From<isize> for DimExpr {
    fn from(value: isize) -> Self {
        DimExpr::Number(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Token {
    Number(isize),
    Variable(String),
    Operator(BinOp),
    LeftParen,
    RightParen,
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' => {
                chars.next(); // Skip whitespace
            }
            '+' => {
                tokens.push(Token::Operator(BinOp::Add));
                chars.next();
            }
            '-' => {
                tokens.push(Token::Operator(BinOp::Sub));
                chars.next();
            }
            '*' => {
                tokens.push(Token::Operator(BinOp::Mul));
                chars.next();
            }
            '/' => {
                tokens.push(Token::Operator(BinOp::Div));
                chars.next();
            }
            '^' => {
                tokens.push(Token::Operator(BinOp::Exp));
                chars.next();
            }
            '(' => {
                tokens.push(Token::LeftParen);
                chars.next();
            }
            ')' => {
                tokens.push(Token::RightParen);
                chars.next();
            }
            '0'..='9' => {
                let mut number = String::new();
                while let Some(&digit) = chars.peek() {
                    if digit.is_numeric() {
                        number.push(digit);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Number(number.parse().map_err(|_| "Invalid number")?));
            }
            'a'..='z' | 'A'..='Z' => {
                let mut var = String::new();
                while let Some(&alpha) = chars.peek() {
                    if alpha.is_alphanumeric() {
                        var.push(alpha);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Variable(var));
            }
            _ => return Err(format!("Unexpected character: {}", ch)),
        }
    }

    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn parse_expr(&mut self) -> Result<DimExpr, String> {
        self.parse_add_sub()
    }

    fn parse_add_sub(&mut self) -> Result<DimExpr, String> {
        let mut node = self.parse_mul_div()?;
        while let Some(Token::Operator(op)) = self.peek().cloned() {
            if matches!(op, BinOp::Add | BinOp::Sub) {
                self.advance();
                let rhs = self.parse_mul_div()?;
                node = DimExpr::Binary(op, Box::new(node), Box::new(rhs));
            } else {
                break;
            }
        }
        Ok(node)
    }

    fn parse_mul_div(&mut self) -> Result<DimExpr, String> {
        let mut node = self.parse_exp()?;
        while let Some(Token::Operator(op)) = self.peek().cloned() {
            if matches!(op, BinOp::Mul | BinOp::Div) {
                self.advance();
                let rhs = self.parse_exp()?;
                node = DimExpr::Binary(op, Box::new(node), Box::new(rhs));
            } else {
                break;
            }
        }
        Ok(node)
    }

    fn parse_exp(&mut self) -> Result<DimExpr, String> {
        let mut node = self.parse_factor()?;
        while let Some(Token::Operator(BinOp::Exp)) = self.peek() {
            self.advance();
            let rhs = self.parse_factor()?;
            node = DimExpr::Binary(BinOp::Exp, Box::new(node), Box::new(rhs));
        }
        Ok(node)
    }

    fn parse_factor(&mut self) -> Result<DimExpr, String> {
        match self.peek().cloned() {
            Some(Token::Number(n)) => {
                self.advance();
                Ok(DimExpr::Number(n))
            }
            Some(Token::Variable(var)) => {
                self.advance();
                Ok(DimExpr::Variable(var.clone()))
            }
            Some(Token::LeftParen) => {
                self.advance();
                let expr = self.parse_expr()?;
                if let Some(Token::RightParen) = self.peek() {
                    self.advance();
                    Ok(expr)
                } else {
                    Err("Expected closing parenthesis".to_string())
                }
            }
            Some(Token::Operator(BinOp::Sub)) => {
                self.advance();
                let expr = self.parse_factor()?;
                Ok(DimExpr::Unary(UnaryOp::USub, Box::new(expr)))
            }
            _ => Err("Unexpected token".to_string()),
        }
    }
}

// Implementing operator overloading for DimExpr

impl Add for DimExpr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        DimExpr::Binary(BinOp::Add, Box::new(self), Box::new(rhs))
    }
}

// Implement Sub operator
impl Sub for DimExpr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        DimExpr::Binary(BinOp::Sub, Box::new(self), Box::new(rhs))
    }
}

// Implement Mul operator
impl Mul for DimExpr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        DimExpr::Binary(BinOp::Mul, Box::new(self), Box::new(rhs))
    }
}

impl Mul<isize> for DimExpr {
    type Output = Self;
    fn mul(self, rhs: isize) -> Self {
        DimExpr::Binary(BinOp::Mul, Box::new(self), Box::new(DimExpr::Number(rhs)))
    }
}

// Implement Div operator
impl Div for DimExpr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        DimExpr::Binary(BinOp::Div, Box::new(self), Box::new(rhs))
    }
}
