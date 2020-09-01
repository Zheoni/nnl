from sly import Parser
from nnl.lexer import NNLexer
import nnl.neural_network as nn


class NNParser(Parser):
    tokens = NNLexer.tokens
    # debugfile = 'parser.out'

    @_('program')
    def S(self, p):
        return p.program
    
    @_('stmt')
    def program(self, p):
        return [p.stmt]
    @_('stmt program')
    def program(self, p):
        return [p.stmt] + p.program

    @_('network')
    def stmt(self, p):
        return p.network
    @_('instruction')
    def stmt(self, p):
        return p.instruction

    @_('input')
    def instruction(self, p):
        return p.input
    @_('output')
    def instruction(self, p):
        return p.output
    @_('run')
    def instruction(self, p):
        return p.run
    
    @_('NAME "{" neuron_properties "}"')
    def neuron(self, p):
        props_list = [prop[1] for prop in p.neuron_properties]

        props_names = [prop[0] for prop in props_list]
        if len(props_names) != len(set(props_names)):
            raise SyntaxError('Cannot have duplicate properties in a neuron')
        del props_names

        props = {prop[0]: prop[1] for prop in props_list}
        
        if 'input' not in props or 'output' not in props:
            raise SyntaxError('All neurons have to define an input and output')
            
        return nn.Neuron(p.NAME, **props)

    @_('neuron_property')
    def neuron_properties(self, p):
        return [p.neuron_property]
    @_('neuron_property "," neuron_properties')
    def neuron_properties(self, p):
        return [p.neuron_property] + p.neuron_properties

    @_('property_definition')
    def neuron_property(self, p):
        return p.property_definition

    @_('prop_input',
    'prop_output',
    'prop_bias',
    'prop_activation')
    def property_definition(self, p):
        return p

    @_('INPUT ":" "[" input_list "]"')
    def prop_input(self, p):
        return (p.INPUT, p.input_list)
    
    @_('OUTPUT ":" "[" output_list "]"')
    def prop_output(self, p):
        return (p.OUTPUT, p.output_list)

    @_('BIAS ":" NUMBER')
    def prop_bias(self, p):
        return (p.BIAS, p.NUMBER)

    @_('ACTIVATION ":" ACTIVATION_FUNCTION')
    def prop_activation(self, p):
        return (p.ACTIVATION, p.ACTIVATION_FUNCTION)

    @_('connection')
    def input_list(self, p):
        return [p.connection]
    @_('connection "," input_list')
    def input_list(self, p):
        return [p.connection] + p.input_list

    @_('NAME ":" NUMBER')
    def connection(self, p):
        return nn.Connection(name=p.NAME, weight=p.NUMBER)

    @_('NAME')
    def output_list(self, p):
        return [p.NAME]
    @_('NAME "," output_list')
    def output_list(self, p):
        return [p.NAME] + p.output_list

    @_('layer_element')
    def layer_elements(self, p):
        return [p.layer_element]
    @_('layer_element "," layer_elements')
    def layer_elements(self, p):
        return [p.layer_element] + p.layer_elements

    @_('network')
    def layer_element(self, p):
        return p.network
    @_('neuron')
    def layer_element(self, p):
        return p.neuron

    @_('NAME "[" layer_elements "]"')
    def layer(self, p):
        return nn.Layer(p.NAME, p.layer_elements)

    @_('network')
    def networks(self, p):
        return [p.network]
    @_('network "," networks')
    def networks(self, p):
        return [p.network] + p.networks

    @_('layer')
    def layers(self, p):
        return [p.layer]
    @_('layer "," layers')
    def layers(self, p):
        return [p.layer] + p.layers
    
    @_('NAME "(" network_content ")"')
    def network(self, p):
        return nn.Network(p.NAME, p.network_content) 

    @_('network "," networks')
    def network_content(self, p):
        return [p.network] + p.networks
    @_('layer "," layers')
    def network_content(self, p):
        return [p.layer] + p.layers

    @_('NAME "=" NUMBER')
    def input(self, p):
        return nn.Instruction('input', name=p.NAME, number=p.NUMBER)

    @_('">" output_value')
    def output(self, p):
        return nn.Instruction('output', name=p.output_value)

    @_('NAME')
    def output_value(self, p):
        return p.NAME
    @_('ALL')
    def output_value(self, p):
        return p.ALL

    @_('"-" ">" NAME')
    def run(self, p):
        return nn.Instruction('run', name=p.NAME)

    def error(self, p):
        raise SyntaxError(f'Syntax error at line {p.lineno}')
        

if __name__ == '__main__':
    lexer = NNLexer()
    parser = NNParser()

    with open("example.nn") as file:
        try:
            text = file.read()
            program = parser.parse(lexer.tokenize(text))
            print(program)
        except EOFError:
            pass
