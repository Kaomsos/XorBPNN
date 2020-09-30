# %%
# print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# from sklearn import datasets
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
# %%
'''
a test fot XorBPNN
'''
from XorBPNN import XorBPNN
from visualize import visualize_scatter, visualize_contour
from numpy import exp
fig_scale = 2

def d_tanh (y_hat):
    return 1 - y_hat ** 2
def tanh(x):
    return (exp(x) - exp(-x))/(exp(x) + exp(-x))
def ReLu(x):
    return x if x > 0 else 0
def d_ReLu(y_hat):
    return 1 if y_hat > 0 else 0
def ELU(x, alpha = 1):
    return alpha * (exp(x) - 1) if x < 0 else x
def d_ELU(y_hat, alpha = 1):
    return y_hat + alpha if y_hat < 0 else 1
def s_tanh(x, s = 1):
    return s * (exp(x) - exp(-x))/(exp(x) + exp(-x))
def d_s_tanh(y_hat, s = 1):
    return s * (1 - y_hat ** 2)

nn = XorBPNN(epoch = 1000, 
                batchsize=4, 
                learning_rate=0.05, 
                hidden_layer_size=2,
                activate=s_tanh,
                d_activate_to_x=d_s_tanh)
# plot initial nn
visualize_scatter(regressor=nn.predict, scale = fig_scale)

# %%
print("the initial weights of BPNN is \n W_1:\n{0}\n W_2:\n{1}".format(nn.W_1, nn.W_2))
X = nn.test_X
y = nn.test_y
print("predition:")
nn.predict(X)

nn.fit(X, y)
print("the trained weights of BPNN is \n W_1:\n{0}\n W_2:\n{1}".format(nn.W_1, nn.W_2))

print("score /accuracy of the NN is: {0}".format(nn.score(X, y)))
print("loss of the nn is:{0}".format(nn.loss(X, y)))
# plot trained nn
visualize_scatter(regressor=nn.predict, scale = fig_scale)
# %%
'''
a test for AssertionError
'''
# for i in range(10):
#             for j in range(5):
#                 print("print({0},{1})".format(i, j))
#             try:
#                 assert i < 5, "Perfect Prediction, Early Stop Triggered."
#             except AssertionError as e:
#                 print(e.args[0])
#                 break
# %%
'''
a test for shuffle
'''
# import numpy as np
# from random import shuffle
# test_X = np.array([[0, 0],
#                     [0, 1],
#                     [1, 0],
#                     [1, 1]])
# test_y = np.array([0,
#                     1,
#                     1,
#                     0])

# test_set = list(zip(test_X, test_y))
# test_set
# %%
'''
shuffled test_set
'''
# shuffle(test_set)
# test_set
# %%
'''
unzip the 
'''
# X, y = zip(*test_set)
# print(X)
# print(y)
# %%
'''
a test for np.meshgrid()
'''
# import numpy as np
# x = np.linspace(-5, 5, 5)
# y = np.linspace(-5, 5, 5)
# i, j = np.meshgrid(x, y)

# %%
