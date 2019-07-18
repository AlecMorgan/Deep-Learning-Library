"""
Optimizers adjust network parameters
based on the gradients computed during
backpropagation, thus minimizing error.
"""
from neural_net import NeuralNet


class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic gradient descent optimizer.
    """
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad