"""
Loss functions measure the quality of a model's prediction.
This enables model optimization by providing a measure of 
what is better and what is worse.
"""

import numpy as np

from tensor import Tensor

# Abstract base class for loss type implementations.
class Loss: 
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """Evaluates a model by MSE (Mean Squared Error)."""
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)