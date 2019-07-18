"""
XOR is the canonical example of a function that 
cannot be learned by a linear model because
it is not linearly separable.
"""
import numpy as np 
from lib.train import train
from lib.neural_net import NeuralNet
from lib.layers import Linear, Tanh

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0], 
    [0, 1],
    [0, 1],
    [1, 0]
])


net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)