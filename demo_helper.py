"""
Helper functions for demo notebooks.
"""
from typing import List
import numpy as np
from math import atan, sin, pi, sqrt
import matplotlib.pyplot as plt

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
    
    
def plot_activation_function(func, func_prime, plot_scale, tangent_points=[]):
    """
    Plots a function in two subplots, one with and one without tangent lines.
    
    Parameters
    ----------
    func : callable
        The function being plotted.
    func_prime : callable
        Function that returns the derivative of a given point for this function.
    plot_scale : float or int
        How large to make the plot/how far to extend away from 0 on the x axis.
    tangent_points : iterable, optional
        The x coordinantes at which to draw each tangent line.
    """
    plt.style.use("dark_background")
    
    # Plotting our activation function
    X = np.linspace(-1 * plot_scale, plot_scale)
    y = func(X)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5 * plot_scale, 10))
    # TODO(Alec): Fix ylim setting bug.
    plt.ylim(-1, 1)
    axes[0].plot(X, y, color="red")
    plt.ylim(-3, 3)
    axes[1].plot(X, y, color="red")

    # Plotting our tangent lines
    for x in tangent_points:
        # Each tangent point is the x coord of a tangent line
        x = np.asarray(x)
        y = func(x)
        m = func_prime(x)
        plot_tangent_lines(axes[1], x, y, m)


def plot_tangent_lines(ax, x, y, m, line_scale=.5):
    """
    Plot a tangent line given an (x, y) point and a slope.
    """
    
    # Radians of all three corners
    print(x, m)
    angle_c = pi / 2
    angle_a = atan(m)
    angle_b = angle_c - angle_a

    # Lengths of all three sides
    c = line_scale
    a = c * sin(angle_a)
    b = sqrt(c**2 - a**2)

    # Coordinates of points on each line
    tangent_x = [x - b, x + b]
    tangent_y = [y - a, y + a]
    ax.plot(tangent_x, tangent_y, color="blue");