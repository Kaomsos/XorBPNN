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
from visualize import visualize

nn = XorBPNN(epoch = 100, 
                batchsize=4, 
                learning_rate=0.1, 
                hidden_layer_size=2)
# plot initial nn
visualize(regressor=nn.predict)

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
visualize(regressor=nn.predict)

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
