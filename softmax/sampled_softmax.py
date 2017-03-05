import numpy as np
import time

# hand-written softmax auto derivative

class SoftmaxLayer:
    def __init__(self, init_W, init_b, n_channels=None):
        self.W = init_W
        self.b = init_b
        if n_channels:
            self.n_channels = n_channels
        else:
            self.n_channels = self.b.shape[0]
    
    def __call__(self, x, y=None):
        W, b, n_channels = self.W, self.b, self.n_channels

        if y is not None:
            assert(isinstance(y, int))
            all_channels = range(self.b.shape[0])
            all_channels.remove(y)
            np.random.shuffle(all_channels)
            channels = [y]
            channels.extend(all_channels[:n_channels-1])
            channels = np.array(channels, dtype=np.int)
            W = W[:, channels]
            b = b[channels]
            y = 0

        s = self.softmax(np.dot(x, W) + b, y)

        if y is not None:
            self.dw = np.tile(x.reshape(1,-1), (W.shape[1], 1))
            self.sliced_delta_W = np.outer(x, self.ds)
            self.delta_W = np.zeros_like(self.W)
            for i in range(n_channels):
                self.delta_W[:, channels[i]] = self.sliced_delta_W[:, i]

        return s

    def softmax(self, x, y=None):
        ex = np.exp(x)
        denom = np.sum(ex)
        a = ex / denom

        # back propagation
        if y is not None:
            assert(isinstance(y, int))
            n = x.shape[0]
            self.ds = np.ones(n, dtype=np.float) * a[y]
            self.ds[y] -= 1.0
        return a



W_1 = np.random.rand(10, 10) * 1e-4
b_1 = np.zeros(10, dtype=np.float)
softmax = SoftmaxLayer(W_1, b_1, 2)

def gen_xy():
    x = np.random.rand(10)
    y = int(np.argmax(x))
    return x, y



# test process
def test_model():
    correct = 0
    n_test = 100
    for t in range(n_test):
        x, y = gen_xy()
        a = softmax(x)
        correct += 1 if np.argmax(a) == y else 0

    print 'accuracy:', 1.0 * correct / n_test

# training process
for t in range(100000):
    x, y = gen_xy()
    a = softmax(x, y)
    softmax.W -= softmax.delta_W * 0.002
    if t % 1000 == 0:
        test_model()