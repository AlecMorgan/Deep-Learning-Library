"""
Helper functions for demo notebooks.
"""
from typing import List

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


def plot_tangent_lines(ax, x, y, m, line_scale=.5):
    # Radians of all three corners
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