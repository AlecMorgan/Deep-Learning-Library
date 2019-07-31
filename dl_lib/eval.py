"""
A small testing suite with support for multi-label prediction.

TODO(Alec): Complete and validate all functions. 
"""
import numpy as np
from dl_lib.tensor import Tensor
from dl_lib.neural_net import NeuralNet


class IllegalArgumentError(ValueError):
    pass

# Accuracy: (tp + tn) / (tp + tn + fp + fn)
# Precision: (tp) / (tp + fp)
# Recall: (tp) / (tp + fn)

def score_accuracy(targets: Tensor,
                   predictions: Tensor, 
                   mode: str = "classification",
                   thres: float = 0.5) -> float:
    """
    Evaluate the neural network's accuracy performance. 
    """
    if mode == "classification":
        predictions = (predictions > thres).astype(int)
        score = (predictions == targets).sum()
    elif mode == "regression":
        raise NotImplementedError
    else: 
        raise IllegalArgumentError
    return score
    
def score_precision(targets: Tensor,
                    predictions: Tensor, 
                    mode: str = "classification",
                    thres: float = 0.5) -> float:
    """
    Evaluate thee neural network's precision performance.
    """
    raise NotImplementedError

def score_recall(targets: Tensor,
                 predictions: Tensor, 
                 mode: str = "classification",
                 thres: float = 0.5) -> float:
    """
    Evaluate the neural network's recall performance.
    """
    raise NotImplementedError