import qwerty_ast
import ast
import readline
from convert_ast import convert_to_qwerty

while True:
    cmd = input('(qwerty) ')
    qwerty = convert_to_qwerty(ast.parse(cmd, mode='single'))
    print(qwerty)