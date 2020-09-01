import abc
from sys import stderr
from typing import List, Tuple, Union, Set, Dict
import nnl.activation_functions as af
from collections import namedtuple


Connection = namedtuple('Connection', ['name', 'weight'])


class Element:
    ...


class Program:
    ...


class VariableStore:
    def __init__(self):
        self._store = {}

    def assign(self, name: str, value: Union[Element, float]) -> None:
        if isinstance(self._store.get(name, None), Element):
            raise ValueError(
                f'Element {name} cannot be overwritten')
        self._store[name] = value

    def add_recursive(self, element: Element) -> None:
        self.assign(element.name, element)
        for sub_element in element.inner_elements():
            self.add_recursive(sub_element)

    def get(self, name: str) -> Union[Element, float]:
        return self._store.get(name, None)

    def has(self, name: str) -> bool:
        return name in self._store


class Statement(abc.ABC):
    @abc.abstractmethod
    def run(self, program: Program) -> None:
        pass


class Element(Statement):
    def __init__(self, name: str, elements: List[Element] = None):
        self.name = name
        self.elements = elements if elements is not None else []

    def inner_elements(self, recursive=False) -> List[Element]:
        if recursive:
            elements = []
            for element in self.elements:
                if isinstance(element, Neuron):
                    elements.append(element)
                else:
                    elements.extend(element.inner_elements(recursive))
            return elements
        else:
            return self.elements

    def contains(self, element: Element) -> bool:
        if element in self.inner_elements():
            return True
        else:
            for sub_element in self.inner_elements():
                if sub_element.contains(element):
                    return True
        return False

    @abc.abstractmethod
    def run(self, program: Program) -> List[float]:
        pass


class Neuron(Element):
    def __init__(self,
                 name: str,
                 input: List[Connection],
                 output: List[str],
                 bias: float = 0.0,
                 activation: str = 'identity'):
        super().__init__(name)
        self.value: float = None
        self.input = input
        self.input_values: List[float] = []
        self.output = output
        self.bias = bias
        self.activation_name = activation
        if activation in af.__dict__:
            self.activation = af.__dict__[activation]
        else:
            raise NotImplementedError(
                f'Activation function "{activation}" is not implemented')

    def run(self, program: Program) -> List[float]:
        self.value = self.bias
        self.input_values.clear()
        for name, weight in self.input:
            variable = program.memory.get(name)
            if variable is None:
                raise RuntimeError(
                    f'Neuron "{self.name}" han a connection to an unknown name')
            elif isinstance(variable, Neuron):
                input_value = variable.value
            elif type(variable) is float:
                input_value = variable
            else:
                raise RuntimeError(
                    f'Neuron "{self.name}" has a connection not to a neuron or an input value')
            self.input_values.append(input_value)
            self.value += input_value * weight
        self.value = self.activation(self.value)
        for out in self.output:
            try:
                program.memory.assign(out, self.value)
            except ValueError:
                pass
        if program.output_all:
            print(f'{self.name}={self.value}')
        return [self.value]

    def pretty_string(self) -> str:
        calcs = ''
        for conn, value in zip(self.input, self.input_values):
            calcs += f'{value} x {conn.weight} + '

        calcs += str(self.bias)

        if self.activation_name != 'identity':
            calcs = f'{self.activation_name}({calcs})'

        ps = f'{self.name} => {calcs} = {self.value}'

        return ps


class Layer(Element):
    def __init__(self, name: str, elements: List[Element]):
        super().__init__(name, elements)

        if any([isinstance(element, Layer) for element in elements]):
            raise ValueError('A layer cannot directly contain another layer')

        if len(elements) <= 0:
            raise ValueError('Cannot create an empty layer')

    def run(self, program: Program) -> List[float]:
        layer_results: List[float] = []
        for element in self.elements:
            layer_results += element.run(program)
        return layer_results

    def final_neurons(self) -> List[Neuron]:
        neurons = []
        for element in self.elements:
            if isinstance(element, Neuron):
                neurons.append(element)
            else:
                neurons.extend(element.final_neurons())
        return neurons

    def initial_neurons(self) -> List[Neuron]:
        neurons = []
        for element in self.elements:
            if isinstance(element, Neuron):
                neurons.append(element)
            else:
                neurons.extend(element.initial_neurons())
        return neurons


class Network(Element):
    def __init__(self, name: str, elements: List[Element]):
        super().__init__(name, elements)
        n_elements = len(elements)
        if n_elements < 1:
            raise ValueError('Cannot create an empty network')

        network_type = type(elements[0])

        if issubclass(network_type, Neuron):
            raise ValueError('Networks cannot directly contain neurons')

        if not all([isinstance(element, network_type) for element in elements]):
            raise ValueError(
                'Cannot mix different element types inside a network')
        if issubclass(network_type, Layer) and n_elements < 2:
            raise ValueError('A network of layers needs at least 2 layers')

    def run(self, program: Program) -> List[float]:
        for element in self.elements[0:-1]:
            element.run(program)
        return self.elements[-1].run(program)

    def _layer_at_index(self, index: int) -> Layer:
        element = self.elements[index]
        if not isinstance(element, Layer):
            element = element._layer_at_index(index)
        return element

    def last_layer(self) -> Layer:
        return self._layer_at_index(-1)

    def first_layer(self) -> Layer:
        return self._layer_at_index(0)

    def final_neurons(self) -> List[Neuron]:
        return self.last_layer().final_neurons()

    def initial_neurons(self) -> List[Neuron]:
        return self.first_layer().initial_neurons()

    def element_before(self, element: Element) -> Element:
        try:
            idx = self.elements.index(element)
            if idx == 0:
                return None
            else:
                return self.elements[idx - 1]
        except (ValueError, IndexError):
            return None


class Instruction(Statement):
    @staticmethod
    def _run(name: str, program: Program) -> None:
        if 'all' in program.outputs:
            program.output_all = True
            program.outputs.remove('all')

        if program.forced_output_all:
            program.output_all = True

        if program.output_all:
            print('=' * 10)

        element = program.memory.get(name)
        if not isinstance(element, Element):
            raise ValueError(f'{name} is not a runnable name')
        if element not in program.elements:
            raise ValueError(f'{name} is not a name in the top level')
        element.run(program)

        if program.output_all:
            print('=' * 10)

        if len(program.outputs) > 0:
            for output in program.outputs:
                value = program.memory.get(output)
                if isinstance(value, Neuron):
                    print(value.pretty_string())
                elif isinstance(value, (Network, Layer)):
                    print(value.name + ':')
                    for neuron in value.final_neurons():
                        print('\t' + neuron.pretty_string())
                elif type(value) is float:
                    print(f'{output}={value}')
                else:
                    print(f'{output}=None')
        else:
            for neuron in element.final_neurons():
                if len(neuron.output) == 1:
                    print(
                        f'{neuron.output[0] if len(neuron.output) == 1 else neuron.name}={neuron.value}')
        program.outputs.clear()
        program.output_all = False

    @staticmethod
    def _input(name: str, number: float, program: Program) -> None:
        if name not in program.forced_inputs:
            program.memory.assign(name, number)

    @staticmethod
    def _output(name: str, program: Program) -> None:
        program.outputs.append(name)

    _instructions = {
        'run': ['name'],
        'input': ['name', 'number'],
        'output': ['name']
    }

    def __init__(self, instruction: str, **args: dict):
        required_arguments = self._instructions.get(instruction, None)

        if required_arguments is None:
            raise ValueError(f'Unknown instruction: "{instruction}"')

        for argument in required_arguments:
            if argument not in args:
                raise ValueError(
                    f'Instruction "{instruction} requires {required_arguments}')

        self.function = Instruction.__dict__['_' + instruction].__func__
        self.args = args

    def run(self, program: Program) -> None:
        self.function(program=program, **self.args)


class Program:
    def __init__(self, statements: List[Statement], forced_inputs: Dict[str, float] = None, show_all: bool = False):
        self.instructions: List[Instruction] = []
        self.elements: List[Element] = []
        self.memory: VariableStore = VariableStore()
        self.inputs: Set[str] = set()
        self.forced_inputs: Set[str] = set()
        self.outputs: List[str] = []
        self.output_all: bool = False
        self.forced_output_all: bool = show_all

        if forced_inputs:
            for name, value in forced_inputs.items():
                self.add_forced_input(name, value)

        for stmt in statements:
            if isinstance(stmt, Element):
                self.elements.append(stmt)
                self.memory.add_recursive(stmt)
            elif isinstance(stmt, Instruction):
                self.instructions.append(stmt)
                if stmt.function is Instruction._input:
                    self.inputs.add(stmt.args['name'])

        if not self._check_connections_logic():
            raise RuntimeError('Neuron conections are not logic')

    def _check_element_logic(self, element: Element, stack: List[Element], internal_outputs: Set[str], alone_internal_outputs: dict) -> bool:
        if isinstance(element, Neuron):
            current_layer: Layer = stack[-1]
            current_network: Network = stack[-2]

            if not isinstance(current_layer, Layer) or not isinstance(current_network, Network):
                raise RuntimeError('Invalid structure')

            layer_before = current_network.element_before(current_layer)
            stack_index = -2
            while layer_before is None and stack_index > -len(stack):
                stack_index -= 1

                inner_element = stack[stack_index + 1]
                parent_element = stack[stack_index]
                if isinstance(parent_element, Network):
                    layer_before = parent_element.element_before(inner_element)

            for input_name, _ in element.input:
                # Cannot be itself
                if input_name == element.name:
                    raise RuntimeError(
                        f'Neuron "{element.name}" cannot have itself as an input')
                value = self.memory.get(input_name)

                # If its a variable
                if value is None or type(value) is float:
                    # Its an input
                    if input_name in self.inputs:
                        # Have to be in the input layer
                        if element not in stack[0].initial_neurons():
                            raise RuntimeError(
                                f'Input value "{input_name}" used outside the input layer on neuron "{element.name}"')
                        # Have to be the only one in there
                        if len(element.input) > 1:
                            raise RuntimeError(
                                f'Neuron "{element.name}" in input layer cannot have more than one input')
                        continue
                    # If not, have to be in the output of something
                    if input_name not in internal_outputs:
                        raise RuntimeError(
                            f'Input "{input_name}" used in neuron "{element.name}" is undefined')
                    elif len(alone_internal_outputs) > 0:
                        # If the variables is used, discard it from the set
                        for neuron, outputs in alone_internal_outputs.items():
                            if input_name in outputs:
                                del alone_internal_outputs[neuron]
                                break
                    # And that something needs to be in the layer before
                    if input_name not in [output for neuron in layer_before.final_neurons() for output in neuron.output]:
                        raise RuntimeError(
                            f'Input "{input_name}" used in neuron "{element.name}" is a temporal variable creating a bad connection between neurons')
                    continue
                # Its an actual neuron
                elif isinstance(value, Neuron):
                    # Cannot be in the input layer
                    if layer_before is None:
                        raise RuntimeError(
                            f'Neuron "{element.name}" in input layer cannot have an input from other neuron')

                    # Have to be in the output of the other one
                    if element.name not in value.output:
                        raise RuntimeError(
                            f'"{element.name}" not in output of "{value.name}"')

                    # Cannot be in the same layer directly
                    if value in current_layer.inner_elements():
                        raise RuntimeError(
                            f'Input "{input_name}" of "{element.name}" is in the same layer')

                    # Need to be in the layer before
                    if layer_before is not None and value not in layer_before.final_neurons():
                        raise RuntimeError(
                            f'Bad connection of input "{input_name}" of neuron "{element.name}"')

                # Its neither of them
                else:
                    raise RuntimeError(
                        f'Input "{input_name}" of "{element.name}" is neither a variable or other neuron')

            for output_name in element.output:
                # Cannot be itself
                if output_name == element.name:
                    raise RuntimeError(
                        f'Neuron "{element.name}" cannot have itself as an output')

                # Cannot be an input
                if output_name in self.inputs:
                    raise RuntimeError(f'An output ("{output_name}") cannot have the same name as an input variable.')

                value = self.memory.get(output_name)
                # If its a variable
                if value is None:
                    internal_outputs.add(output_name)

                    # if there are only variables in the output and its not in the output layer add it to the alone variables set
                    if not any([isinstance(self.memory.get(out), Neuron) for out in element.output]) and element not in stack[0].final_neurons():
                        if element in alone_internal_outputs:
                            alone_internal_outputs[element].append(output_name)
                        else:
                            alone_internal_outputs[element] = [output_name]

                    neurons_in_current_layer = current_layer.inner_elements(
                        recursive=True)
                    neurons_in_current_layer.remove(element)
                    # Cannot be overwritten in the same layer because the connections wont work
                    if output_name in [output for neuron in neurons_in_current_layer for output in neuron.output]:
                        raise RuntimeError(
                            f'Variable "{output_name}" written more than once in the same layer')
                # ita a neuron
                elif isinstance(value, Neuron):
                    # Needs to be in the input of the other one
                    if element.name not in [conn.name for conn in value.input]:
                        raise RuntimeError(
                            f'Output "{output_name}" of neuron "{element.name}" not in input of "{value.name}"')
                # neither
                else:
                    raise RuntimeError(
                        f'Output "{output_name}" of neuron "{element.name}" is neither a variable or other neuron')
        
        elif isinstance(element, Layer) or isinstance(element, Network):
            for sub_element in element.inner_elements():
                stack.append(element)
                if not self._check_element_logic(sub_element, stack, internal_outputs, alone_internal_outputs):
                    return False
                stack.pop()
        return True

    def _check_connections_logic(self) -> bool:
        for element in self.elements:
            stack = []
            internal_outputs = set()
            alone_internal_outputs = {}
            correct = self._check_element_logic(element, stack, internal_outputs, alone_internal_outputs)
            
            # If this is true, means that a neuron that only outputted to variables, some of those variables are never used
            if len(alone_internal_outputs) != 0:
                raise RuntimeError(f'Internal variable(s) is declared as the only output of a neuron but never used: {list(alone_internal_outputs.values())}')

            if not correct:
                return False

        return True

    def run(self) -> None:
        for instruction in self.instructions:
            instruction.run(self)

    def add_forced_input(self, name: str, value: float) -> None:
        self.memory.assign(name, value)
        self.inputs.add(name)
        self.forced_inputs.add(name)
