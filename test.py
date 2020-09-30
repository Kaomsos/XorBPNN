# %%
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# %%
for i in range(10):
            for j in range(5):
                print("print({0},{1})".format(i, j))
            try:
                assert i < 5, "Perfect Prediction, Early Stop Triggered."
            except AssertionError as e:
                print(e.args[0])
                break
# %%
