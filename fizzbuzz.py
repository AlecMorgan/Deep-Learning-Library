"""
Demo: solving FizzBuzz with an MLP (Multi-Layer Perceptron)

FizzBuzz is the following problem:

For each of the numbers 1 to 100:
* if the number is divisible by 3, print "fizz"
* if the number is divisible by 5, print "buzz"
* if the number is divisible by 15, print "fizzbuzz"
* otherwise, just print the number
"""

import numpy as np 
from typing import List
from dl_lib.train import train
from dl_lib.neural_net import NeuralNet
from dl_lib.layers import Linear, Tanh
from dl_lib.optim import SGD

def fizz_buzz_encode(x: int) -> List[int]:
    """
    Return a categorical encoding of a correct FizzBuzz output.
    """
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def binary_encode(x: int) -> List[int]:
    """
    10 digit binary encoding of x
    """
    return [x >> i & 1 for i in range(10)]

# Mapping. We expect to see targets[i] given inputs[i].
inputs = np.array([
    binary_encode(x)
    for x in range(101, 1024)
])

# Mapping. We expect to see targets[i] given inputs[i].
targets = np.array([
    fizz_buzz_encode(x)
    for x in range(101, 1024)
])

# Simple MLP net. 
net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])

train(net,
      inputs,
      targets,
      num_epochs=5000,
      optimizer=SGD(lr=0.001))

# Testing FizzBuzz net on values from 1 to 100.
for x in range(1, 101):
    predicted = net.forward(binary_encode(x))
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    print(x, labels[predicted_idx], labels[actual_idx])