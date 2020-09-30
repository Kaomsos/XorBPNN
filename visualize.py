from matplotlib import pyplot as plt
import numpy as np
from numpy import random
from IPython.display import display # <- where the display() is included

## see through its prediction
def visualize_scatter(regressor, N = 10000, scale = 1, threshhold = 0.5):
    '''
    2-dimensional inputs -> regressor -> array-like output
    scatter with pyplot
    '''
    X = (random.rand(N) * scale) - scale / 2 + 0.5
    Y = (random.rand(N) * scale) - scale / 2 + 0.5
    points = list(zip(X, Y))
    output = regressor(points)
    color = list(map(lambda x: 'r' if x >= threshhold else 'b', output))
    # plot
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.vlines(1, 0, 1, colors = 'black', linewidth = 3)
    ax.vlines(0, 0, 1, colors = 'black', linewidth = 3)
    ax.hlines(1, 0, 1, colors = 'black', linewidth = 3)
    ax.hlines(0, 0, 1, colors = 'black', linewidth = 3)

    ax.scatter(X, Y, c=color, alpha = 0.6)
    ax.scatter([1, 0], [1, 0], color = 'b', s = 100, linewidth = 1, edgecolors = 'black')
    ax.scatter([1, 0], [0, 1], color = 'r', s = 100, linewidth = 1, edgecolors = 'black')
    # display(fig)

def visualize_contour(regressor, scale_xy = 1):
    x = np.linspace(-scale_xy/2, scale_xy/2, scale_xy * 50)
    y = np.linspace(-scale_xy/2, scale_xy/2, scale_xy * 50)
    X, Y = np.meshgrid(x, y)
    points = np.dstack((X, Y))
    Z = []
    for r in points:
        Z.append(regressor(r))
    Z = np.array(Z)


    fig = plt.figure(figsize = (10, 10))
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.contour3D(X, Y, np.ones((scale_xy * 50, scale_xy * 50)) * 0.5, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
