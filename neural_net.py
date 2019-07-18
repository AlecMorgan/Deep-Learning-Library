"""
Neural networks are composed of multiple layers,
each of which is essentially a tensor of values.
Different layers will learn different weights
and biases which collectively form the intelli-
gence of the neural network. 
"""
from typing import Sequence, Iterator, Tuple
from tensor import Tensor
from layers import Layer, Activation


class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            if isinstance(layer, Activation) is not True:
                for name, param in layer.params.items():
                    grad = layer.grads[name]
                    yield param, grad