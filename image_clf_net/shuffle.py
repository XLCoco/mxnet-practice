import os
import numpy as np
import random

train_loc = r"$data_folder\train"


def make_dogvcat_data(data_loc, set_name):

    # FORMAT ::= int_image_index \t label_index \t path_to_image \n

    image_files = os.listdir(data_loc)
    random.seed(100)
    random.shuffle(image_files)

    n_image = len(image_files)
    n_train = int(n_image * 0.8)
    n_test = n_image - n_train

    # train
    fout = open(os.path.join(data_loc+'\\..', set_name+'_train.lst'), 'w')

    DOG = 0
    CAT = 1

    for i in range(n_train):
        filename = image_files[i]
        label = DOG if 'dog' in filename else CAT
        fout.write('%d\t%d\t%s\n'%(i, label, filename))

    fout.close()

    # test
    fout = open(os.path.join(data_loc+'\\..', set_name+'_test.lst'), 'w')

    for i in range(n_test):
        filename = image_files[n_train+i]
        label = DOG if 'dog' in filename else CAT
        fout.write('%d\t%d\t%s\n'%(i, label, filename))

    fout.close()

make_dogvcat_data(train_loc, "cats_vs_dogs")