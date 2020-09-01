from nnl.lexer import NNLexer
from nnl.parser import NNParser
from nnl.neural_network import Program

from sys import argv
import traceback


def run_file(filename: str, inputs: dict = None, show_all: bool = False):
    lexer = NNLexer()
    parser = NNParser()

    try:
        with open(filename) as file:
            try:
                text = file.read()
                statements = parser.parse(lexer.tokenize(text))
                program = Program(statements,
                                  forced_inputs=inputs,
                                  show_all=show_all)
                program.run()
            except EOFError:
                pass
    except OSError:
        print('Cannot open the file')
    except SyntaxError as e:
        print(e)
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()
