
import numpy as np
import time

# hand-written softmax auto derivative

class SoftmaxLayer:
    def __init__(self, init_W, init_b):
        self.W = init_W
        self.b = init_b
    
    def __call__(self, x):
        W, b = self.W, self.b
        self.dw = np.tile(x.reshape(1,-1).T, W.shape[1])
        s = self.softmax(np.dot(x, W) + b)
        delta_W = self.dw * self.ds
        print 'dW: \n', delta_W
        print 'db: \n', self.ds
        return s

    def softmax(self, x, bp=False):
        ex = np.exp(x)
        denom = np.sum(ex)
        a = ex / denom
        self.ds = a - a**2
        return a

        

W = np.array([[1., 2., 3.], [1., 1., 2.], [1., 0., 1.]])
b = np.array([0., 0., 0.])
x = np.array([1., 2., 3.]) * 0.1
softmax = SoftmaxLayer(W, b)
print softmax(x)
