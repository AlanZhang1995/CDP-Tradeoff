import numpy as np

import os
import urllib
import gzip
import cPickle as pickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(images, targets, batch_size, shuffle=1):
    if shuffle==1:
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(targets)

    def get_epoch(shuffle=1):
        if shuffle==1:
            rng_state = np.random.get_state()
            np.random.shuffle(images)
            np.random.set_state(rng_state)
            np.random.shuffle(targets)
        '''
        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])
        '''
        image_batches = images.reshape(-1, batch_size, 3072*2)
        target_batches = targets.reshape(-1, batch_size, 10)

        for i in xrange(len(image_batches)):
            yield (np.copy(image_batches[i]), np.copy(target_batches[i]))

    return get_epoch

'''
def load(batch_size, data_dir):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir), 
        cifar_generator(['test_batch'], batch_size, data_dir)
    )
'''
def load(filepath, train_batch_size, valid_batch_size, test_batch_size):
    #load data
    r = np.load(filepath)
    train_data=r['train']
    test_data=r['test']

    r = np.load(filepath.replace('LRMnist','Label'))
    train_label=r['train']
    test_label=r['test']
    
    #select validation
    train_data_tmp = []
    train_label_tmp = []
    dev_data = []
    dev_label = []
    select_list=np.random.permutation(len(train_data))
    select_list=select_list[:5000]
    for i in range(len(train_data)):
        if i in select_list:
            dev_data.append(train_data[i])
            dev_label.append(train_label[i])
        else:
            train_data_tmp.append(train_data[i])
            train_label_tmp.append(train_label[i])
    dev_data=np.array(dev_data)
    dev_label=np.array(dev_label)
    train_data=np.array(train_data_tmp)
    train_label=np.array(train_label_tmp)
    print('Check::',dev_data.shape,dev_label.shape,train_data.shape,train_label.shape)

    return (
        cifar_generator(train_data, train_label, train_batch_size), 
        cifar_generator(dev_data, dev_label, valid_batch_size, shuffle=0), 
        cifar_generator(test_data, test_label, test_batch_size, shuffle=0)
    )