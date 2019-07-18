"""
Demo: solving XOR with an MLP (Multi-Layer Perceptron)

XOR is the canonical example of a function that 
cannot be learned by a linear model because
it is not linearly separable.
"""
import numpy as np 
from dl_lib.train import train
from dl_lib.neural_net import NeuralNet
from dl_lib.layers import Linear, Tanh

# Mapping. We expect to see targets[i] given inputs[i].
inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

# Mapping. We expect to see targets[i] given inputs[i].
targets = np.array([
    [1, 0], 
    [0, 1],
    [0, 1],
    [1, 0]
])


# Simple MLP net.
net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)