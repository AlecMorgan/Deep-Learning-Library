"""
Trains neural networks.
"""
from dl_lib.tensor import Tensor
from dl_lib.neural_net import NeuralNet
from dl_lib.loss import Loss, MSE
from dl_lib.optim import Optimizer, SGD
from dl_lib.data import DataIterator, BatchIterator
from tqdm import tqdm


def train(net: NeuralNet, 
          inputs: Tensor, 
          targets: Tensor,
          num_epochs: int = 5000, 
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(), 
          optimizer: Optimizer = SGD()) -> None:
    """
    Train the neural network on the inputs/targets data. 
    """
    epoch_losses = []
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        epoch_losses.append(epoch_loss)
    return epoch_losses