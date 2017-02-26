import os
import sys
import time
import numpy as np
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt


def one_hot(i, n):
    a = np.zeros(n)
    a[i] = 1
    return a


def make_data(root, name):
    
    alphabet = 'ABCDEFGHIJ'
    
    data = []
    label = []
    
    n_classes = len(alphabet)

    print 'n_classes: %d' % n_classes
    print 'starting extraction...'
    print ''

    n_total = 0
    n_failed = 0

    original_shape = None
    flattened_len = 0

    start_time = time.time()

    for i, c in enumerate(alphabet):
        folder = os.path.join(root, c)
        file_count = len(os.listdir(folder))

        sys.stdout.write('\b\b\r')
        sys.stdout.write('Extracting from folder ' + c + '\n')
        sys.stdout.write('%d/%d\n'%(0, file_count))
        sys.stdout.flush()

        for j, filename in enumerate(os.listdir(folder)):
            sys.stdout.write('\b\r')
            sys.stdout.write('%d/%d'%(j+1, file_count))
            sys.stdout.flush()

            n_total += 1
            img_file = os.path.join(folder, filename)
            
            try:
                img_data = ndimage.imread(img_file).astype(float)
                if not original_shape: original_shape = img_data.shape
                img_data = img_data.reshape(-1,) # flattened
                if flattened_len == 0: flattened_len = img_data.shape[0]
                assert(img_data.shape[0] == flattened_len)
            except Exception: 
                n_failed += 1
                continue
            else: # learn to use try-else
                data.append(img_data)
                label.append(one_hot(i, n_classes))
    
    dataset = np.hstack((np.array(data), np.array(label)))
    np.save(name, dataset)
    
    n_samples = n_total - n_failed

    print 'the dataset has been built successfully!'
    print 'time elapsed: %.2f sec' % (time.time() - start_time)
    print '---' 
    print 'sample original shape: ', original_shape
    print 'flattened length: %d' % flattened_len
    print 'files total: %d' % n_total
    print 'failed: %d'      % n_failed
    print '#samples: %d'    % n_samples

    fout = open(name+'.desc', 'w')
    fout.write('n_samples:%d\n'%n_samples)
    fout.write('n_classes:%d\n'%n_classes)
    fout.write('flattened_length:%d\n'%flattened_len)
    fout.write('original_shape:%s\n'%str(original_shape))
    fout.close()

    return (n_classes, n_samples, flattened_len, original_shape)


class DataIter:
    def __init__(self, dataset):
        self.dataset = np.load(dataset, mmap_mode='r')
        
    def __getitem__(self, index):
        return self.dataset[index]


if __name__ == '__main__':
    root = r'D:\dataset\notMNIST_small'
    data_file_name = 'notMNIST'
    n_classes, n_samples, flattened_shape, original_shape = make_data(root, data_file_name)

    dataset = data_file_name + '.npy'
    data = load_partial(np.random.randint(500, 600, 10), dataset)

    print data.shape

    samples = data[:, :-n_classes]
    labels = data[:, -n_classes:]

    imgplot = plt.imshow(samples[0].reshape(28, 28))
    plt.show()

