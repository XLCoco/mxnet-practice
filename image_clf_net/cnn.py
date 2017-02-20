import mxnet as mx
import numpy as np

import logging

logging.getLogger().setLevel(logging.DEBUG)


def get_lenet():

    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(4,4), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))

    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh4 = mx.symbol.Activation(data=fc1, act_type="tanh")

    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh4, num_hidden=2)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet


batch_size = 100

model = mx.model.FeedForward(
    ctx = mx.gpu(0),
    symbol = get_lenet(),
    num_epoch = 50,
    learning_rate = 0.01,
)

train_iter = mx.io.ImageRecordIter(
    path_imgrec="data/train.rec",
    data_shape=(3,32,32),
    batch_size=batch_size,
)

val_iter = mx.io.ImageRecordIter(
    path_imgrec="data/test.rec",
    data_shape=(3,32,32),
    batch_size=batch_size,
)

model.fit(
    X = train_iter,
    eval_data = val_iter,
    batch_end_callback = mx.callback.Speedometer(batch_size, 200)
)

print 'score on test set:', model.score(val_iter)
