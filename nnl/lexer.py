from sly import Lexer

name_re = r'[A-Z][a-zA-Z0-9]*'
number_re = r'(-?(([1-9][0-9]+)|[0-9]))(\.[0-9]+)?((e|E)(\+|-)?[0-9]+)?'

class NNLexer(Lexer):
    tokens = {
        ACTIVATION_FUNCTION,
        INPUT,
        OUTPUT,
        BIAS,
        ACTIVATION,
        NUMBER,
        NAME,
        ALL
    }

    ignore = ' \t'
    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += t.value.count('\n')

    literals = {
        '{', '}',
        '[', ']',
        '(', ')', 
        '+', '-',
        '>', ':',
        '=', ','
    }

    ACTIVATION_FUNCTION = r'sigmoid|identity|binary'

    INPUT = r'input'
    OUTPUT = r'output'
    BIAS = r'bias'
    ACTIVATION = r'activation'
    ALL = r'all'

    @_(number_re)
    def NUMBER(self, t):
        t.value = float(t.value)
        return t

    NAME = name_re

    def error(self, t):
        print(f'Bad character "{t.value[0]}" at line "{self.lineno}"')
        self.index += 1


if __name__ == "__main__":
	with open("example.nn") as file:
		data = file.read()
		lexer = NNLexer()
		for tok in lexer.tokenize(data):
			print(tok)
