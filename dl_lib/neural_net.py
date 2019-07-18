"""
Neural networks are composed of multiple layers,
each of which is essentially a tensor of values.
Different layers will learn different weights
and biases which collectively form the intelli-
gence of the neural network. 
"""
from typing import Sequence, Iterator, Tuple
from dl_lib.tensor import Tensor
from dl_lib.layers import Layer, Activation


class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward-propagate inputs through layers.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """
        Yield weight/bias parameters and their error gradients for
        all layers throughout the neural network. 
        """
        for layer in self.layers:
            if isinstance(layer, Activation) is not True:
                for name, param in layer.params.items():
                    grad = layer.grads[name]
                    yield param, grad