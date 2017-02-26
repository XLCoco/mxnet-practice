
import numpy as np
from datahelper import DataIter
from scipy import ndimage
from sklearn.linear_model import LogisticRegression, SGDClassifier
from matplotlib import pyplot as plt
import cPickle as pickle

data_iter = DataIter('notMNIST.npy')
n_classes, n_samples, flattened_shape, original_shape = 10, 18724, 784, (28, 28)

indexes = np.random.permutation(range(n_samples))
n_train = int(0.8 * n_samples)
n_test = n_samples - n_train

ind_train = indexes[:n_train]
ind_test = indexes[-n_test:]


# init model
model = LogisticRegression(solver='lbfgs', multi_class='multinomial', warm_start=True)
# model = SGDClassifier()

# model test method
test_data = data_iter[ind_test]
def test_model():
    x_test = test_data[:, :-n_classes]
    y_test = np.argmax(test_data[:, -n_classes:], axis=1).reshape(-1)

    y_pred = model.predict(x_test).reshape(-1)
    correct = (y_pred == y_test).astype(np.int)

    return 'accuracy: %.2f%%' % (np.sum(correct) * 1.0 / n_test)


# train model by batch
def train_sgd(batch_size):
    n_batches = n_train // batch_size
    for i in range(n_batches):
        start, end = i * batch_size, (i+1) * batch_size
        batch = data_iter[ind_train[start:end]]
        samples = batch[:, :-n_classes]
        labels = np.argmax(batch[:, -n_classes:], axis=1)

        if samples.shape[0] == 0:
            break

        model.partial_fit(samples, labels, classes=range(n_classes))
        
        if i % 10 == 0: print 'batch %d: '%i, test_model()

        with open('model_sgd.pk', 'wb') as f:
            pickle.dump(model, f)
        
    print test_model()


# train model by all data
import time
def train_gd():

    start = time.time()

    data = data_iter[ind_train]
    samples = data[:, :-n_classes]
    labels = np.argmax(data[:, -n_classes:], axis=1)
    model.fit(samples, labels)
    
    print 'model trained in %.2f seconds'%(time.time() - start)
    
    with open('model_gd.pk', 'wb') as f:
        pickle.dump(model, f)

    print test_model()


train_gd()



for i in range(10):
    x = test_data[i, :-n_classes]
    print chr(model.predict(x)[0] + ord('A'))
    plt.imshow(x.reshape(28, 28))
    plt.show()
