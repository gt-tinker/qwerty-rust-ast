"""
Convert a Python AST to a Qwerty AST by recognizing patterns in the Python AST
formed by Qwerty syntax.
"""

import ast
import qwerty_ast
#################### COMMON CODE FOR BOTH @QPU AND @CLASSICAL DSLs ####################


def convert_to_qwerty(py_ast):
    print(ast.dump(py_ast, indent=4))
    if not isinstance(py_ast, (ast.Interactive, ast.Module, ast.Expression)):
        return "Invalid AST"
    
    value = py_ast.body[0]
    if not isinstance(value, ast.Expr):
        return "Invalid syntax; expected an expression"
    
    try:
        return convert_expr(value.value)
    except Exception as e:
        return f"Error: {e}"
    
#   1) check if the instance is a binOp, if so, check its left or right values and then perform the binOp. only sum is implemented for now
#   2) check if the instance is a constant and return it as a tensor, return value in a tensor

    
def convert_expr(node):
    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, str) and all(num in "01" for num in val):
            tensor = qwerty_ast.NodeBox.new_qubit_tensor(node.value)
            # print("tensor type:", type(tensor))
            # print("tensor contents:", dir(tensor))            
            return tensor
        else:
            raise ValueError("Invalid character, e.g. not 1s and 0s")
        
    elif isinstance(node, ast.BinOp):
        left = convert_expr(node.left)
        right = convert_expr(node.right)
        if isinstance(node.op, ast.Add):
            return concat_tensors(left, right)
        else:
            raise NotImplementedError("Operation is not supported, only addition for now")
        
    # elif isinstance(node, ast.Name):
    #     symbol_table[node.id] = qwerty_ast.NodeBox.resolve_name(node.id)
    #     return qwerty_ast.NodeBox.resolve_name(node.id)
    # else:
    #     raise NotImplementedError("Variable not supported")

def concat_tensors(left, right):
    left_val = left.get_tensor()
    right_val = right.get_tensor()
    combined = str(left_val) + str(right_val)
    # print(combined)
    return qwerty_ast.NodeBox.new_qubit_tensor(combined)

# def unpack_ast(ast):
#     try:
#         return ast.body[0].value.value
#     except AttributeError:
#         return None



# def convert_ast(module: ast.Module, filename: str = '', line_offset: int = 0,
#                 col_offset: int = 0) -> Tuple[Kernel, List[bool]]:
#     """
#     Take in a Python AST for a function parsed with ``ast.parse(mode='exec')``
#     and return a ``Kernel`` Qwerty AST node.
#
#     A list of bools is also returned, where each entry in the list corresponds
#     1-to-1 with a dimension variables and is true if and only if the dimension
#     variable was specified explicitly with the syntax e.g. ``@qpu[[N(3)]]`` (to
#     set ``N=3``). For example, with ``@qpu[[I,J(5),K]]``, this function would
#     return ``[false,true,false]``.
#
#     The ``line_offset`` and `col_offset`` are useful (respectively) because a
#     ``@qpu``/``@classical`` kernel may begin after the first line of the file,
#     and the caller may de-indent source code to avoid angering ``ast.parse()``.
#     """
#     return convert_qpu_ast(module, filename, line_offset, col_offset)
#


    

# class BaseVisitor:
#     """
#     Common Python AST visitor for both ``@classical`` and ``@qpu`` kernels.
#     """
#
#     def __init__(self, filename: str = '', line_offset: int = 0, col_offset: int = 0, no_pyframe: bool = False):
#         """
#         Constructor. The ``no_pyframe`` flag is used by the tests to avoid
#         including frames (see ``errs.py``) in DebugInfos constructed by
#         ``get_debug_info()`` below, since this complicates testing.
#         """
#         self.filename = filename
#         self.line_offset = line_offset
#         self.col_offset = col_offset
#         # TODO: Figure out a way to pass this data around instead of storing it
#         #       in this member variable and setting it inside certain visitors
#         self.dim_vars = set()
#         self.frame = None if no_pyframe else _get_frame()
#
#     def get_node_row_col(self, node: ast.AST):
#         if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
#             row = node.lineno + self.line_offset
#             col = node.col_offset + 1 + self.col_offset
#             return row, col
#         else:
#             return None, None
#
#     def get_debug_info(self, node: ast.AST) -> DebugInfo:
#         """
#         Extract line and column number from a Python AST node and return a
#         Qwerty DebugInfo instance.
#         """
#         row, col = self.get_node_row_col(node)
#         return DebugInfo(self.filename, row or 0, col or 0, self.frame)
#
#     def visit_List(self, nodes: List[ast.AST]):
#         """
#         Convenience function to visit each node in a ``list`` and return the
#         results of each as a new list.
#         """
#         return [self.visit(node) for node in nodes]
#
#     def base_visit(self, node: ast.AST):
#         """
#         Convert a Python AST node into a Qwerty AST Node (and return the
#         latter).
#         """
#         if isinstance(node, list):
#             return self.visit_List(node)
#         else:
#             node_name = type(node).__name__
#             raise QwertySyntaxError(f'Unknown Python AST node {node_name}',
#                                     self.get_debug_info(node))
#
# #################### @QPU DSL ####################
#
# # NOTE: For now, users should be able to do either +- or pm
# STATE_CHAR_MAPPING = {
#     '+': (PLUS, X),
#     '-': (MINUS, X),
#     'p': (PLUS, X),
#     'm': (MINUS, X),
#     'i': (PLUS, Y),
#     'j': (MINUS, Y),
#     '0': (PLUS, Z),
#     '1': (MINUS, Z),
# }
#
# EMBEDDING_KINDS = (
#     EMBED_XOR,
#     EMBED_SIGN,
#     EMBED_INPLACE,
# )
#
# EMBEDDING_KEYWORDS = {embedFailed to authenticate. ding_kind_name(e): e
#                       for e in EMBEDDING_KINDS}
#
# RESERVED_KEYWORDS = {'id', 'discard', 'discardz', 'measure', 'flip'}
#
# class QpuVisitor(BaseVisitor):
#     """
#     Python AST visitor for syntax specific to ``@qpu`` kernels.
#     """
#
#     def __init__(self, filename: str = '', line_offset: int = 0,
#                  col_offset: int = 0, no_pyframe: bool = False):
#         super().__init__(filename, line_offset, col_offset, no_pyframe)
#
#     def extract_float_expr(self, node: ast.AST):
#         """
#         Extract a float expression, like a tilt, for example::
#
#             '1' @ (3*pi/2)
#                    ^^^^^^
#         """
#         dbg = self.get_debug_info(node)
#
#         if isinstance(node, ast.Name) and node.id == 'pi':
#             return FloatLiteral(dbg, math.pi)
#         if isinstance(node, ast.Name) and node.id in self.dim_vars:
#             return FloatDimVarExpr(dbg, self.extract_dimvar_expr(node))
#         elif isinstance(node, ast.Constant) \
#                 and type(node.value) in (int, float):
#             return FloatLiteral(dbg, float(node.value))
#         elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
#             return FloatNeg(dbg, self.extract_float_expr(node.operand))
#         elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
#             return FloatBinaryOp(dbg, FLOAT_DIV,
#                                  self.extract_float_expr(node.left),
#                                  self.extract_float_expr(node.right))
#         elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
#             return FloatBinaryOp(dbg, FLOAT_MUL,
#                                  self.extract_float_expr(node.left),
#                                  self.extract_float_expr(node.right))
#         elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
#             return FloatBinaryOp(dbg, FLOAT_POW,
#                                  self.extract_float_expr(node.left),
#                                  self.extract_float_expr(node.right))
#         elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
#             return FloatBinaryOp(dbg, FLOAT_MOD,
#                                  self.extract_float_expr(node.left),
#                                  self.extract_float_expr(node.right))
#         elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
#             return FloatBinaryOp(dbg, FLOAT_ADD,
#                                  self.extract_float_expr(node.left),
#                                  self.extract_float_expr(node.right))
#         elif isinstance(node, ast.Name):
#             return self.visit(node)
#         elif isinstance(node, ast.Subscript) \
#                 and isinstance(node.value, ast.Name) \
#                 and isinstance(node.slice, ast.List) \
#                 and len(node.slice.elts) == 1:
#             val = self.visit(node.value)
#             idx = node.slice.elts[0]
#             lower = self.extract_dimvar_expr(idx)
#             upper = lower.copy()
#             upper += DimVarExpr('', 1)
#             return Slice(dbg, val, lower, upper)
#         else:
#             node_name = type(node).__name__
#             raise QwertySyntaxError('Unsupported float expression {}'
#                                     .format(node_name),
#                                     self.get_debug_info(node))
#
#     def visit_FunctionDef(self, func_def: ast.FunctionDef) -> Kernel:
#         """
#         Convert a ``@qpu`` kernel into a ``QpuKernel`` Qwerty AST node.
#         """
#         return super().base_visit_FunctionDef(func_def, 'qpu', AST_QPU)
#
#     # Variable or reserved keyword
#     def visit_Name(self, name: ast.Name):
#         """
#         Convert a Python AST identitifer node into either a Qwerty primitive
#         AST node or a Qwerty variable name AST node. For example ``id`` becomes
#         an ``Identity`` AST node, and ``foobar`` becomes a Qwerty ``Variable``
#         AST node.
#         """
#         var_name = name.id
#         dbg = self.get_debug_info(name)
#
#         if var_name == 'id':
#             return Identity(dbg, DimVarExpr('', 1))
#         elif var_name == 'std':
#             return BuiltinBasis(dbg, Z, DimVarExpr('', 1))
#         elif var_name == 'pm':
#             return BuiltinBasis(dbg, X, DimVarExpr('', 1))
#         elif var_name == 'ij':
#             return BuiltinBasis(dbg, Y, DimVarExpr('', 1))
#         elif var_name == 'discard':
#             return Discard(dbg, DimVarExpr('', 1))
#         elif var_name == 'measure':
#             # Sugar for `std.measure'
#             return Measure(dbg, BuiltinBasis(dbg.copy(), Z, DimVarExpr('', 1)))
#         elif var_name == 'flip':
#             # Sugar for `std.flip'
#             return Flip(dbg, BuiltinBasis(dbg.copy(), Z, DimVarExpr('', 1)))
#         elif var_name == 'fourier':
#             raise QwertySyntaxError('fourier is a reserved keyword. The '
#                                     'one-dimensional Fourier basis must be '
#                                     'written as fourier[1]',
#                                     self.get_debug_info(name))
#         else:
#             return Variable(dbg, var_name)
#
#     def visit_Constant(self, const: ast.Constant):
#         """
#         Convert a Python string literal into a Qwerty ``QubitLiteral`` AST
#         node. Since the ``QubitLiteral`` Qwerty AST node supports only one
#         eigenstate and primitive basis, this produces a tree of ``BiTensor``s
#         concatenating multiple ``QubitLiteral`` nodes as necessary.
#         For example, ``'0011'`` becomes a ``BiTensor`` of two different
#         ``QubitLiteral`` nodes.
#         """
#         value = const.value
#         if isinstance(value, str):
#             state_chars = value
#             if not state_chars:
#                 raise QwertySyntaxError('Qubit literal must not be an empty string',
#                                         self.get_debug_info(const))
#
#             result = None
#             last_char, last_dim = None, None
#
#             def add_to_tensor_tree():
#                 nonlocal result, last_char, last_dim
#
#                 dbg = self.get_debug_info(const)
#                 eigenstate, prim_basis = STATE_CHAR_MAPPING[last_char]
#                 vec = QubitLiteral(dbg, eigenstate, prim_basis, last_dim)
#
#                 if result is None:
#                     result = vec
#                 else:
#                     # Need to make a second DebugInfo since the last one is going
#                     # to get std::move()'d away by the QubitLiteral constructor above
#                     dbg = self.get_debug_info(const)
#                     result = BiTensor(dbg, result, vec)
#
#             for i, c in enumerate(state_chars):
#                 if c not in STATE_CHAR_MAPPING:
#                     raise QwertySyntaxError('Unknown state |{}⟩ in qubit literal'
#                                             .format(c),
#                                             self.get_debug_info(const))
#                 if last_char == c:
#                     last_dim += DimVarExpr('', 1)
#                 else:
#                     if last_char is not None:
#                         add_to_tensor_tree()
#                     last_char = c
#                     last_dim = DimVarExpr('', 1)
#
#                 if i+1 == len(state_chars):
#                     add_to_tensor_tree()
#
#             return result
#         else:
#             raise QwertySyntaxError('Unknown constant syntax',
#                                     self.get_debug_info(const))
#
#     # Broadcast tensor, i.e., tensoring something with itself repeatedly
#     def visit_Subscript(self, subscript: ast.Subscript):
#         """
#         Convert a Python getitem expression into a Qwerty ``BroadcastTensor``
#         AST node. For example, the Python syntax ``t[N]`` becomes a
#         ``BroadcastTensor`` with a child that is the conversion of ``t``.
#         There is special-case handling here for ``fourier[N]`` and any other
#         inseparable bases added in the future.
#         """
#         value, slice_ = subscript.value, subscript.slice
#         dbg = self.get_debug_info(subscript)
#
#         if isinstance(list_ := slice_, ast.List) and list_.elts:
#             value_node = self.visit(value)
#             instance_vals = [self.extract_dimvar_expr(elt) for elt in list_.elts]
#             return Instantiate(dbg, value_node, instance_vals)
#
#         factor = self.extract_dimvar_expr(slice_)
#
#         # Special case: Fourier basis
#         if isinstance(value, ast.Name) \
#                 and value.id == 'fourier':
#             return BuiltinBasis(dbg, FOURIER, factor)
#
#         value_node = self.visit(value)
#         return BroadcastTensor(dbg, value_node, factor)
#
#     def visit_BinOp(self, binOp: ast.BinOp):
#         if isinstance(binOp.op, ast.Add):
#             return self.visit_BinOp_Add(binOp)
#         elif isinstance(binOp.op, ast.BitOr):
#             return self.visit_BinOp_BitOr(binOp)
#         elif isinstance(binOp.op, ast.RShift):
#             return self.visit_BinOp_RShift(binOp)
#         elif isinstance(binOp.op, ast.BitAnd):
#             return self.visit_BinOp_BitAnd(binOp)
#         elif isinstance(binOp.op, ast.MatMult):
#             return self.visit_BinOp_MatMult(binOp)
#         else:
#             op_name = type(binOp.op).__name__
#             raise QwertySyntaxError('Unknown binary operation {}'
#                                     .format(op_name),
#                                     self.get_debug_info(binOp))
#
#     def visit_BinOp_Add(self, binOp: ast.BinOp):
#         """
#         Convert a Python binary add expression into a Qwerty tensor product AST
#         node. For example, ``t1 + t2`` becomes a ``BiTensor`` Qwerty AST node
#         with two children.
#         """
#         left = self.visit(binOp.left)
#         right = self.visit(binOp.right)
#         dbg = self.get_debug_info(binOp)
#         return BiTensor(dbg, left, right)
#
#     def visit_BinOp_BitOr(self, binOp: ast.BinOp):
#         """
#         Convert a Python bitwise OR expression into a Qwerty ``Pipe`` (function
#         call) AST node. For example, ``t1 | t2`` becomes a ``Pipe`` node with
#         two children.
#         """
#         left = self.visit(binOp.left)
#         right = self.visit(binOp.right)
#         dbg = self.get_debug_info(binOp)
#         return Pipe(dbg, left, right)
#
#     def visit_BinOp_BitAnd(self, binOp: ast.BinOp):
#         """
#         Convert a Python bitwise AND expression into a Qwerty ``Pred``
#         (predication) AST node. For example, ``t1 & t2`` becomes a ``Pred``
#         node with two children — one should be a basis and one should be a
#         function.
#         """
#         basis = self.visit(binOp.left)
#         body = self.visit(binOp.right)
#         dbg = self.get_debug_info(binOp)
#         return Pred(dbg, basis, body)
#
#     def visit_BinOp_MatMult(self, binOp: ast.BinOp):
#         """
#         Convert a Python matrix multiplication expression into a Qwerty
#         ``Phase`` (tilt) AST node. For example, ``t1 @ t2`` becomes a ``Phase``
#         node with two children — the left should be a rev_qfunc[N] or qubit[N],
#         and the right should be a float. If the right operand is in degrees
#         (this is the default unless you write ``t1 @ rad(t2)``), then a
#         conversion to radians is automatically synthesized.
#         """
#         if isinstance(call := binOp.right, ast.Call) \
#                 and isinstance(name := call.func, ast.Name) \
#                 and name.id in ('deg', 'rad'):
#             unit = name.id
#             if call.keywords:
#                 raise QwertySyntaxError(
#                     'Keyword arguments not supported for {}(...)'.format(unit),
#                     self.get_debug_info(binOp))
#             if len(call.args) != 1:
#                 raise QwertySyntaxError(
#                     'Wrong number of arguments {} != 1 passed to {}(...)'
#                     .format(len(call.args), unit),
#                     self.get_debug_info(binOp))
#             angle = call.args[0]
#             # Set the debug location for the angle expression code to this
#             # pseudo-function (deg() or rad())
#             angle_conv_dbg_node = call
#         else:
#             unit = 'deg'
#             angle = binOp.right
#             angle_conv_dbg_node = binOp.right
#
#         angle_expr = self.extract_float_expr(angle)
#         if unit == 'deg':
#             angle_conv_dbg = self.get_debug_info(angle_conv_dbg_node)
#             # Convert to radians: angle_expr/360 * 2*pi
#             # The canonicalizer AST pass will fold this
#             angle_expr = \
#                 FloatBinaryOp(
#                     angle_conv_dbg.copy(),
#                     FLOAT_MUL,
#                     FloatBinaryOp(
#                         angle_conv_dbg.copy(),
#                         FLOAT_DIV,
#                         angle_expr,
#                         FloatLiteral(
#                             angle_conv_dbg.copy(), 360.0)),
#                     FloatLiteral(
#                         angle_conv_dbg.copy(),
#                         2*math.pi))
#
#         dbg = self.get_debug_info(binOp)
#         lhs = self.visit(binOp.left)
#         return Phase(dbg, angle_expr, lhs)
#
#     def visit_BinOp_RShift(self, binOp: ast.BinOp):
#         """
#         Convert a Python right bit shift AST node to a Qwerty
#         ``BasisTranslation`` AST node. For example, ``b1 >> b2`` becomes a
#         ``BasisTranslation`` AST node with two basis children.
#         """
#         basis_in, basis_out = binOp.left, binOp.right
#
#         dbg = self.get_debug_info(binOp)
#         basis_in_node = self.visit(basis_in)
#         basis_out_node = self.visit(basis_out)
#         return BasisTranslation(dbg, basis_in_node, basis_out_node)
#
#     def visit_UnaryOp(self, unaryOp: ast.UnaryOp):
#         if isinstance(unaryOp.op, ast.USub):
#             return self.visit_UnaryOp_USub(unaryOp)
#         elif isinstance(unaryOp.op, ast.Invert):
#             return self.visit_UnaryOp_Invert(unaryOp)
#         else:
#             op_name = type(unaryOp.op).__name__
#             raise QwertySyntaxError('Unknown unary operation {}'
#                                     .format(op_name),
#                                     self.get_debug_info(unaryOp))
#
#     def visit_UnaryOp_USub(self, unaryOp: ast.UnaryOp):
#         """
#         Convert a Python unary negation AST node into a Qwerty AST node tilting
#         the operand by 180 degrees. For example, ``-f`` or ``-'0'``.
#         """
#         dbg = self.get_debug_info(unaryOp)
#         value = self.visit(unaryOp.operand)
#         # Euler's identity, e^{iπ} = -1
#         return Phase(dbg, FloatLiteral(dbg.copy(), math.pi), value)
#
#     def visit_UnaryOp_Invert(self, unaryOp: ast.UnaryOp):
#         """
#         Convert a Python unary bitwise complement AST node into a Qwerty
#         ``Adjoint`` AST node. For example, ``~f`` becomes an ``Adjoint`` node
#         with 1 child (``f``).
#         """
#         unary_operand = unaryOp.operand
#         dbg = self.get_debug_info(unaryOp)
#         operand = self.visit(unary_operand)
#         return Adjoint(dbg, operand)
#
#     def visit_Set(self, set_: ast.Set):
#         """
#         Convert a Python set literal AST node into a Qwerty ``BasisLiteral``
#         AST node. For example, ``{'0', -'1'}`` is a ``BasisLiteral`` node with
#         two children.
#         """
#         dbg = self.get_debug_info(set_)
#         basis_elts = self.visit(set_.elts)
#         return BasisLiteral(dbg, basis_elts)
#
#     def list_comp_helper(self, gen: Union[ast.GeneratorExp, ast.ListComp]):
#         """
#         Helper used by both ``visit_GeneratorExp()`` and ``visit_ListComp()``,
#         since both Python AST nodes have near-identical fields.
#         """
#         dbg = self.get_debug_info(gen)
#
#         if len(gen.generators) != 1:
#             raise QwertySyntaxError('Multiple generators are unsupported in '
#                                     'Qwerty', self.get_debug_info(gen))
#         comp = gen.generators[0]
#
#         if comp.ifs:
#             raise QwertySyntaxError('"if" not supported inside repeat '
#                                     'construct',
#                                     self.get_debug_info(gen))
#         if comp.is_async:
#             raise QwertySyntaxError('async generators not supported',
#                                     self.get_debug_info(gen))
#
#         if not isinstance(comp.target, ast.Name) \
#                 or not isinstance(comp.iter, ast.Call) \
#                 or not isinstance(comp.iter.func, ast.Name) \
#                 or comp.iter.func.id != 'range' \
#                 or len(comp.iter.args) != 1 \
#                 or comp.iter.keywords:
#             raise QwertySyntaxError('Unsupported generator syntax (only '
#                                     'basic "x for i in range(N) is '
#                                     'supported")',
#                                     self.get_debug_info(gen))
#
#         loopvar = comp.target.id
#
#         if loopvar in self.dim_vars:
#             raise QwertySyntaxError('Index variable {} collides with the '
#                                     'name of another type variable'
#                                     .format(loopvar),
#                                     self.get_debug_info(gen))
#         self.dim_vars.add(loopvar)
#         body = self.visit(gen.elt)
#         self.dim_vars.remove(loopvar)
#
#         ub = self.extract_dimvar_expr(comp.iter.args[0])
#
#         return (dbg, body, loopvar, ub)
#
#     def visit_GeneratorExp(self, gen: ast.GeneratorExp):
#         """
#         Convert a Python generator expression AST node into a Qwerty ``Repeat``
#         AST node. For example the highlighted part of the code::
#
#             ... | (func for i in range(20)) | ...
#                    ^^^^^^^^^^^^^^^^^^^^^^^
#
#         is converted to a Repeat AST node. Here, ``range`` is a keyword whose
#         operand is a dimension variable expression.
#         """
#         return Repeat(*self.list_comp_helper(gen))
#
#     def visit_ListComp(self, comp: ast.ListComp):
#         """
#         Convert a Python list comprehension expression AST node into a Qwerty
#         ``RepeatTensor`` AST node. For example, this code:
#
#             ['0' for i in range(5)]
#
#         is converted to a RepeatTensor AST node whose child is a QubitLiteral.
#         (This particular example is equivalent to '00000'.) Here, ``range`` is
#         a keyword whose operand is a dimension variable expression.
#         """
#         return RepeatTensor(*self.list_comp_helper(comp))
#
#     def visit_Call(self, call: ast.Call):
#         """
#         As syntactic sugar, convert a Python call expression into a ``Pipe``
#         Qwerty AST node. In general, the shorthand::
#
#             f(arg1,arg2,...,argn)
#
#         is equivalent to::
#
#             (arg1,arg2,...,argn) | f
#
#         There is also unrelated special handling for ``b.rotate(theta)`` and
#         ``fwd.inplace(rev)``.
#         """
#         if call.keywords:
#             raise QwertySyntaxError('Keyword arguments not supported in '
#                                     'call', self.get_debug_info(call))
#
#         dbg = self.get_debug_info(call)
#
#         # Handling for b.rotate(theta) and fwd.inplace(rev)
#         if isinstance(attr := call.func, ast.Attribute):
#             if attr.attr == 'rotate':
#                 if len(call.args) != 1:
#                     raise QwertySyntaxError('Wrong number of operands '
#                                             '{} != 1 to .rotate'
#                                             .format(len(call.args)),
#                                             self.get_debug_info(attr))
#                 arg = call.args[0]
#                 basis = self.visit(attr.value)
#                 theta = self.extract_float_expr(arg)
#                 return Rotate(dbg, basis, theta)
#             elif attr.attr in EMBEDDING_KEYWORDS \
#                     and isinstance(name := attr.value, ast.Name):
#                 if not call.args:
#                     embedding_kind = EMBEDDING_KEYWORDS[attr.attr]
#                     if embedding_kind_has_operand(embedding_kind):
#                         raise QwertySyntaxError('Keyword {} requires an '
#                                                 'operand'.format(attr.attr),
#                                                 self.get_debug_info(attr))
#                     classical_func_name = name.id
#                     return EmbedClassical(dbg, classical_func_name, '',
#                                           embedding_kind)
#                 else:
#                     embedding_kind = EMBEDDING_KEYWORDS[attr.attr]
#                     if not embedding_kind_has_operand(embedding_kind):
#                         raise QwertySyntaxError('Keyword {} does not require an '
#                                                 'operand'.format(attr.attr),
#                                                 self.get_debug_info(attr))
#
#                     if len(call.args) != 1:
#                         raise QwertySyntaxError('Wrong number of operands '
#                                                 '{} != 1 to {}'
#                                                 .format(len(call.args),
#                                                         attr.attr),
#                                                 self.get_debug_info(attr))
#                     arg = call.args[0]
#                     if not isinstance(arg, ast.Name):
#                         raise QwertySyntaxError('Argument to {} must be an '
#                                                 'identifier, not a {}'
#                                                 .format(attr.attr,
#                                                         type(arg).__name__),
#                                                 self.get_debug_info(attr))
#
#                     classical_func_name = name.id
#                     classical_func_operand_name = arg.id
#                     return EmbedClassical(dbg, classical_func_name,
#                                           classical_func_operand_name,
#                                           embedding_kind)
#
#         rhs = self.visit(call.func)
#         lhs_elts = [self.visit(arg) for arg in call.args]
#         lhs = TupleLiteral(dbg.copy(), lhs_elts) \
#             if len(call.args) != 1 \
#             else self.visit(call.args[0])
#         return Pipe(dbg, lhs, rhs)
#
#     def visit_Tuple(self, tuple_: ast.Tuple):
#         """
#         Convert a Python tuple literal into a Qwerty tuple literal. Trust me,
#         this one is thrilling.
#         """
#         dbg = self.get_debug_info(tuple_)
#         elts = self.visit(tuple_.elts)
#         return TupleLiteral(dbg, elts)
#
#     def visit_Attribute(self, attr: ast.Attribute):
#         """
#         Convert a Python attribute access AST node into Qwerty AST nodes for
#         primitives such as ``.measure`` or ``.flip``. For example,
#         ``std.measure`` becomes a ``Measure`` AST node with a ``BuiltinBasis``
#         child node.
#         """
#         dbg = self.get_debug_info(attr)
#         attr_lhs = attr.value
#         attr_rhs = attr.attr
#
#         if attr_rhs == 'measure':
#             basis = self.visit(attr_lhs)
#             return Measure(dbg, basis)
#         elif attr_rhs == 'project':
#             basis = self.visit(attr_lhs)
#             return Project(dbg, basis)
#         elif attr_rhs == 'q':
#             bits = self.visit(attr_lhs)
#             return Lift(dbg, bits)
#         elif attr_rhs == 'prep':
#             operand = self.visit(attr_lhs)
#             return Prepare(dbg, operand)
#         elif attr_rhs == 'flip':
#             operand = self.visit(attr_lhs)
#             return Flip(dbg, operand)
#         elif attr_rhs in EMBEDDING_KEYWORDS:
#             if isinstance(attr_lhs, ast.Name):
#                 name = attr_lhs
#                 classical_func_name = name.id
#             else:
#                 raise QwertySyntaxError('Keyword {} must be applied to an '
#                                         'identifier, not a {}'
#                                         .format(attr_rhs,
#                                                 type(attr_lhs).__name__),
#                                         self.get_debug_info(attr))
#
#             embedding_kind = EMBEDDING_KEYWORDS[attr_rhs]
#
#             if embedding_kind_has_operand(embedding_kind):
#                 raise QwertySyntaxError('Keyword {} requires an operand, '
#                                         'specified with .{}(...)'
#                                         .format(attr_rhs, attr_rhs),
#                                         self.get_debug_info(attr))
#
#             return EmbedClassical(dbg, classical_func_name, '', embedding_kind)
#         else:
#             raise QwertySyntaxError('Unsupported keyword {}'.format(attr_rhs),
#                                     self.get_debug_info(attr))
#
#     def visit_IfExp(self, ifExp: ast.IfExp):
#         """
#         Convert a Python conditional expression AST node into a Qwerty
#         classical branching AST node. For example, ``x if y or z`` becomes a
#         Qwerty ``Conditional`` AST node with three children.
#         """
#         if_expr, then_expr, else_expr = ifExp.test, ifExp.body, ifExp.orelse
#         dbg = self.get_debug_info(ifExp)
#
#         if_expr_node = self.visit(if_expr)
#         then_expr_node = self.visit(then_expr)
#         else_expr_node = self.visit(else_expr)
#
#         return Conditional(dbg, if_expr_node, then_expr_node, else_expr_node)
#
#     def visit_BoolOp(self, boolOp: ast.BoolOp):
#         if isinstance(boolOp.op, ast.Or):
#             return self.visit_BoolOp_Or(boolOp)
#         else:
#             op_name = type(boolOp.op).__name__
#             raise QwertySyntaxError('Unknown boolean operation {}'
#                                     .format(op_name),
#                                     self.get_debug_info(boolOp))
#
#     def visit_BoolOp_Or(self, boolOp: ast.BoolOp):
#         """
#         Convert a Python Boolean expression with an ``or`` into a Qwerty
#         superposition AST node. For example, ``0.25*'0' or 0.75*'1'`` becomes a
#         ``SuperpositionLiteral`` AST node with two children.
#         """
#         dbg = self.get_debug_info(boolOp)
#         operands = boolOp.values
#         if len(operands) < 2:
#             raise QwertySyntaxError('Superposition needs at least two operands', dbg)
#
#         pairs = []
#         had_prob = False
#
#         for operand in operands:
#             # Common case: 0.5 * '0'
#             if isinstance(mult_binop := operand, ast.BinOp) \
#                     and isinstance(mult_binop.op, ast.Mult):
#                 prob_node = mult_binop.left
#                 vec_node = mult_binop.right
#             # Deal with some operator precedence aggravation: for 0.5 * '0'@45
#             # the root node is actually @. Rearrange this for programmer
#             # convenience
#             elif isinstance(matmult_binop := operand, ast.BinOp) \
#                     and isinstance(matmult_binop.op, ast.MatMult) \
#                     and isinstance(mult_binop := matmult_binop.left, ast.BinOp) \
#                     and isinstance(mult_binop.op, ast.Mult):
#                 prob_node = mult_binop.left
#                 vec_node = ast.BinOp(mult_binop.right, ast.MatMult(),
#                                      matmult_binop.right, lineno=matmult_binop.lineno,
#                                      col_offset=matmult_binop.col_offset)
#             else:
#                 prob_node = None
#                 vec_node = operand
#
#             has_prob = prob_node is not None
#
#             if pairs and has_prob ^ had_prob:
#                 operand_dbg = self.get_debug_info(operand)
#                 raise QwertySyntaxError(
#                     'Either all operands of a superposition operator should '
#                     'have explicit probabilities, or none should have '
#                     'explicit probabilities', operand_dbg)
#
#             had_prob = has_prob
#
#             if has_prob:
#                 if not isinstance(prob_node, ast.Constant):
#                     prob_dbg = self.get_debug_info(prob_node)
#                     raise QwertySyntaxError(
#                         'Currently, probabilities in a superposition literal '
#                         'must be integer constants', prob_dbg)
#                 prob_const_node = prob_node
#                 prob_val = prob_const_node.value
#
#                 if not isinstance(prob_val, float) \
#                         and not isinstance(prob_val, int):
#                     prob_dbg = self.get_debug_info(prob_node)
#                     raise QwertySyntaxError(
#                         'Probabilities in a superposition literal must be '
#                         'floats, not {}'.format(str(type(prob_val))), prob_dbg)
#             else:
#                 prob_val = 1.0 / len(operands)
#
#             pair = (prob_val, self.visit(vec_node))
#             pairs.append(pair)
#
#         return SuperposLiteral(dbg, pairs)
#
#     def visit(self, node: ast.AST):
#         if isinstance(node, ast.Name):
#             return self.visit_Name(node)
#         elif isinstance(node, ast.Constant):
#             return self.visit_Constant(node)
#         elif isinstance(node, ast.Subscript):
#             return self.visit_Subscript(node)
#         elif isinstance(node, ast.BinOp):
#             return self.visit_BinOp(node)
#         elif isinstance(node, ast.UnaryOp):
#             return self.visit_UnaryOp(node)
#         elif isinstance(node, ast.Set):
#             return self.visit_Set(node)
#         elif isinstance(node, ast.GeneratorExp):
#             return self.visit_GeneratorExp(node)
#         elif isinstance(node, ast.ListComp):
#             return self.visit_ListComp(node)
#         elif isinstance(node, ast.Call):
#             return self.visit_Call(node)
#         elif isinstance(node, ast.Tuple):
#             return self.visit_Tuple(node)
#         elif isinstance(node, ast.Attribute):
#             return self.visit_Attribute(node)
#         elif isinstance(node, ast.IfExp):
#             return self.visit_IfExp(node)
#         elif isinstance(node, ast.BoolOp):
#             return self.visit_BoolOp(node)
#         else:
#             return self.base_visit(node)
#
# def convert_qpu_ast(module: ast.Module, filename: str = '', line_offset: int = 0,
#                     col_offset: int = 0) -> Kernel:
#     """
#     Run the ``QpuVisitor`` on the provided Python AST to convert to a Qwerty
#     ``@qpu`` AST and return the result. The return value is the same as
#     ``convert_ast()`` above.
#     """
#     if not isinstance(module, ast.Module):
#         raise QwertySyntaxError('Expected top-level Module node in Python AST',
#                                 None) # This should not happen
#
#     visitor = QpuVisitor(filename, line_offset, col_offset)
#     return visitor.visit_Module(module), visitor.tvs_has_explicit_value
#
# def convert_qpu_expr(expr: ast.Expression, filename: str = '',
#                      line_offset: int = 0, col_offset: int = 0,
#                      no_pyframe: bool = False) -> Expr:
#     """
#     Convert an expression from a @qpu kernel instead of the whole thing.
#     Currently used only in unit tests. Someday could be used in a REPL, for
#     example.
#     """
#     if not isinstance(expr, ast.Expression):
#         raise QwertySyntaxError('Expected top-level Expression node in '
#                                 'Python AST', None) # This should not happen
#
#     visitor = QpuVisitor(filename, line_offset, col_offset, no_pyframe)
#     return visitor.visit_Expression(expr)