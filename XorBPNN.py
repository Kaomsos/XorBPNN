# %%
'''
as simple as importing
'''
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import numpy.random as random
from random import shuffle
from EXC_EarlyStop import EarlyStop

# %%
'''
I define some useful functions in this cell
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

vsigmoid = np.vectorize(sigmoid)

def MSE(y_hat, y):
    return (y_hat - y) ** 2

def pd_MSE_to_y_hat(y_hat, y):
    '''
    2 * (y_hat - y)
    '''
    return 2 * (y_hat - y)

def d_sigmoid_to_x(y_hat):
    '''
    y_hat * (1 - y_hat)
    '''
    return y_hat * (1 - y_hat)

def Sign(x, threshhold = 0.5):
    '''
    return 1 if x >= threshhold
    else return 0
    '''
    return 1 if x >= threshhold else 0

vSign = np.vectorize(Sign)

# %% 
'''
implementation of XorBPNN
'''
class XorBPNN (BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate = 0.1, epoch = 100, batchsize = 1, hidden_layer_size = 2):
        """
        Called when initializing the classifier
        """
        self.hidden_layer_size = hidden_layer_size
        # weights
        ## random initialization
        self.W_1 = random.rand(self.hidden_layer_size, 2 + 1)   # size = 2 * 3
        self.W_2 = random.rand(self.hidden_layer_size + 1)    # size = 1 * 3
        ## zero initialization
        # self.W_1 = np.zeros(shape = (self.hidden_layer_size, 2 + 1))   # size = 2 * 3
        # self.W_2 = np.zeros(shape = self.hidden_layer_size + 1)    # size = 1 * 3
        # layers
        self.input = np.ones(2 + 1)        # size = 3 * 1
        self.hidden_layer = np.ones(self.hidden_layer_size + 1) # size = 3 * 1
        '''
        remember not to change the first "design component" (which equals 1) 
        '''
        self.output = 0                     # number/scalar # which equals y_hat
        # hyper-parameter
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batchsize = batchsize
        # and then initialize the layers
        self.forward(np.ones(2))
        
        # built_in dataset
        self.test_X = np.array([[0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1]])
        self.test_y = np.array([0,
                                1,
                                1,
                                0])
    # def set_params(self, **params):
    #     return super().set_params(**params)
    # def get_params(self, deep=True):
    #     return super().get_params(deep=deep)
    '''
    All estimators must have get_params and set_params functions. 
    They are inherited when you subclass BaseEstimator and 
    I would recommend not to override these function 
    (just not state them in definition of your classifier).
    ''' 
    def fit(self, X, y):
        '''
        Let's kindly ignore exception handeling here
        '''
        '''
        You'll almost surely want add some new attributes to your object 
        which are created in fit method. 
        These should be ended by _ at the end, e.g. self.fitted_
        '''
        for i in range(self.epoch):
            batch = list(zip(self.test_X, self.test_y))
            shuffle(batch)
            batch_X, batch_y = zip(*batch)
            for j in range(self.batchsize):
                self.forward(batch_X[j])
                self.backward(batch_y[j])
            print("fiting...   epoch = {0}, loss = {1}".format(i + 1, self.loss(self.test_X, self.test_y)))
            try:
                assert self.score(self.test_X, self.test_y) < 1, "Perfect Prediction, Early Stop　Triggered."
            except AssertionError as e:
                print(e.args[0])
                break
        return self

    def predict(self, X):
        '''
        ignore the constraint that training is before predicting
        '''
        return [self.forward(x) for x in X]
    
    def score(self, X, y, sample_weight=None):
        '''
        return the accuracy
        '''
        _l = vSign(self.predict(X)) == y
        return sum(_l) / len(_l)
    def loss(self, X, y):
        '''
        compute the average loss over the 4 training data
        provide an insight into the training process 
        '''
        sum = 0
        for e in self.predict(X) - y:
            sum += e ** 2
        return sum / len(y)

    def forward(self, x):
        '''
         take x as input, return the output //
         from input/X to output/y_hat
         and the weights keep fixed
        '''
        self.input[1 : ] = x
        v_h = vsigmoid(np.matmul(self.W_1, self.input))             # size: 2 * 1
        self.hidden_layer[1 : ] = v_h                               # extend to size: 3 * 1
        self.output = sigmoid(np.matmul(self.W_2, self.hidden_layer))   # scalar
        return self.output

    def backward(self, y):
        '''
        take y as input and update the paramenters, no return //
        from output/y_hat to previous weights
        where 'y' is the actual value for a sample 
        '''
        # compute the difference while the neurons are fixed
        common_coef = pd_MSE_to_y_hat(y_hat=self.output, y=y) * d_sigmoid_to_x(y_hat=self.output)
        mat = [[self.W_2[j + 1] * d_sigmoid_to_x(self.hidden_layer[j + 1]) * self.input[i] for i in range(2 + 1)] for j in range(self.hidden_layer_size)]
        diff_W_2 = common_coef * self.hidden_layer
        diff_W_1 = common_coef * np.array(mat)
        # update the weights
        self.W_2 -= self.learning_rate * diff_W_2
        self.W_1 -= self.learning_rate * diff_W_1