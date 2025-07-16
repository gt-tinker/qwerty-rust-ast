from .convert_ast import convert_qpu_repl
import ast
from ._qwerty_pyrt import ReplState

def repl():
    state = ReplState()
    while True:
        cmd = input('(qwerty) ')
        qwerty = convert_qpu_repl(ast.parse(cmd, mode='single'))
        state.run(qwerty)