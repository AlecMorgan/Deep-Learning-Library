"""
Various layer types of neural network layers.
A common feature of all types is that they must
pass inputs forward and propagate gradients 
backward (backpropagation). For example,
a neural network's architecture might
look like:

inputs -> Linear -> Tanh -> Linear -> output
"""

from typing import Dict, Callable
import numpy as np
from dl_lib.tensor import Tensor
from dl_lib.activations import (
    tanh, tanh_prime, relu, relu_prime, 
    leaky_relu, leaky_relu_prime
)


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce outputs corresponding to these inputs.
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate the error gradient and optimize to reduce it.
        """
        raise NotImplementedError


class Linear(Layer):
    """
    computes outputs = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        (where c is some vector)
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    Activation layers simply apply a function
    element-wise to their inputs.
    """
    def __init__(self, f: F, f_prime: F) -> None:
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """ 
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z).
        This is simply the chain rule
        applied element-wise.
        """
        return self.f_prime(self.inputs) * grad

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)

class Relu(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)

class Leaky_Relu(Activation):
    def __init__(self):
        super().__init__(leaky_relu, leaky_relu_prime)

    def forward(self, inputs: Tensor, neg_slope: float = .1) -> Tensor:
        self.inputs = inputs
        return self.f(inputs, neg_slope)

    def backward(self, grad: Tensor, neg_slope: float = .1) -> Tensor:
        """ 
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z).
        This is simply the chain rule
        applied element-wise.
        """
        return self.f_prime(self.inputs, neg_slope) * grad