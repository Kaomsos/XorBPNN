from matplotlib import pyplot as plt
import numpy as np
from numpy import random
from IPython.display import display # <- where the display() is included

## see through its prediction
def visualize(regressor, N = 5000):
    '''
    2-dimensional inputs -> regressor -> array-like output
    scatter with pyplot
    '''
    X = random.rand(N)
    Y = random.rand(N)
    points = list(zip(X, Y))
    output = regressor(points)
    color = list(map(lambda x: 'r' if x >= 0.5 else 'b', output))
    # plot
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.scatter(X, Y, c=color)
    # display(fig)