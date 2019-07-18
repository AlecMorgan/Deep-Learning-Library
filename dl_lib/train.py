"""
Trains neural networks.
"""
from dl_lib.tensor import Tensor
from dl_lib.neural_net import NeuralNet
from dl_lib.loss import Loss, MSE
from dl_lib.optim import Optimizer, SGD
from dl_lib.data import DataIterator, BatchIterator


def train(net: NeuralNet, 
          inputs: Tensor, 
          targets: Tensor,
          num_epochs: int = 5000, 
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(), 
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)