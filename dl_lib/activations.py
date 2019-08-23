from dl_lib.tensor import Tensor
import numpy as np


def tanh(x: Tensor) -> Tensor:
    """
    Apply the hyperbolic tangent function to the
    input tensor element-wise and return the result.
    """
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    """
    Return the derivative of the input's hyperbolic tangent.
    """
    y = tanh(x)
    return 1 - y ** 2


def relu(x: Tensor) -> Tensor:
    """
    Apply the ReLU function to the input
    tensor element-wise and return the result.
    """
    # If the element is negative, it becomes 0.
    # Otherwise it stays the same. Therefore, 
    # taking the maximum of each element and 0
    # is a sufficient implementation.
    return np.maximum(x, 0)


def relu_prime(x: Tensor) -> Tensor:
    """
    Return the derivative of the input's ReLU.
    """
    y = x.astype(float)
    y[y < 0] = 0
    y[y > 0] = 1
    return y


def leaky_relu(x: Tensor, neg_slope: float = .1) -> Tensor: 
    """
    Apply the leaky ReLU function to the input
    tensor element-wise and return the result. 
    Leaky ReLU mitigates the "dying ReLU" problem
    by keeping negative values slightly negative.
    """
    return np.maximum(x, x * neg_slope)


def leaky_relu_prime(x: Tensor, neg_slope: float = .1) -> Tensor:
    """
    Return the derivative of the input's leaky ReLU.
    """
    y = x.astype(float)
    y[y > 0] = 1
    y[y < 0] = neg_slope
    return y


def sigmoid(x: Tensor) -> Tensor:
    """
    Apply the sigmoid function to the input
    tensor element-wise and return the result. 
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: Tensor) -> Tensor:
    """
    Return the derivative of the input's sigmoid.
    """
    y = sigmoid(x)
    return y * (1 - y)