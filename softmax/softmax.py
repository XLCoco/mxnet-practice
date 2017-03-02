
import numpy as np
import time

# hand-written softmax auto derivative

class SoftmaxLayer:
    def __init__(self, init_W, init_b):
        self.W = init_W
        self.b = init_b
    
    def __call__(self, x, y=None):
        W, b = self.W, self.b
        s = self.softmax(np.dot(x, W) + b, y)

        if y is not None:
            self.dw = np.tile(x.reshape(1,-1), (W.shape[1], 1))
            self.delta_W = np.outer(x, self.ds)
            # print 'delta_W: \n', self.delta_W
            # print 'dw: \n', self.dw
            # print 'db: \n', self.ds

        return s

    def softmax(self, x, y=None):
        ex = np.exp(x)
        denom = np.sum(ex)
        a = ex / denom

        # back propagation
        if y is not None:
            n = x.shape[0]
            self.ds = np.ones(n, dtype=np.float) * a[y]
            self.ds[y] -= 1.0
        
        return a

        

W = np.random.rand(3, 2) * 1e-4
b = np.array([0., 0.])
x = np.array([3., 5., 4.])
softmax = SoftmaxLayer(W, b)
h = softmax(x, 0)

for t in range(1000):
    y = softmax(x, 0)
    softmax.W -= softmax.delta_W * 0.01
    if t % 50 == 0:
        print y
