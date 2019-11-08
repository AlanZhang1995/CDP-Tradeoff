import numpy

import os
import urllib
import gzip
import cPickle as pickle

def mnist_generator(images, targets, batch_size, n_labelled, limit=None, shuffle=1):
    #print(images.shape) 
    #print(targets.shape) 
    if shuffle==1:
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)
    if limit is not None:
        print "WARNING ONLY FIRST {} MNIST DIGITS".format(limit)
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch(shuffle=1):
        if shuffle==1:
            rng_state = numpy.random.get_state()
            numpy.random.shuffle(images)
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(targets)

        if n_labelled is not None:
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(labelled)

        image_batches = images.reshape(-1, batch_size, 784*2)
        target_batches = targets.reshape(-1, batch_size, 10)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]), numpy.copy(labelled))

        else:

            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]))

    return get_epoch

'''
def load(batch_size, test_batch_size, n_labelled=None):
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print "Couldn't find MNIST dataset in /tmp, downloading..."
        urllib.urlretrieve(url, filepath)

    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    print(train_data)
    r = numpy.load("NoisyMnist_1.npz")
    train_image=r['train']
    dev_image=r['valid']
    test_image=r['test']

    return (
        mnist_generator(train_data, train_image, batch_size, n_labelled), 
        mnist_generator(dev_data, dev_image, test_batch_size, n_labelled), 
        mnist_generator(test_data, test_image, test_batch_size, n_labelled)
    )
'''
def load(filepath, batch_size, test_batch_size, n_labelled=None):
    r = numpy.load(filepath)
    train_data=r['train']
    dev_data=r['valid']
    test_data=r['test']

    r = numpy.load(filepath.replace('LRMnist','Label'))
    train_label=r['train']
    dev_label=r['valid']
    test_label=r['test']

    return (
        mnist_generator(train_data, train_label, batch_size, n_labelled), 
        mnist_generator(dev_data, dev_label, test_batch_size, n_labelled), 
        mnist_generator(test_data, test_label, test_batch_size, n_labelled)
    )

def load_test(filepath, test_batch_size, n_labelled=None):
    r = numpy.load(filepath)
    test_data=r['test']

    r = numpy.load(filepath.replace('LRMnist','Label'))
    test_label=r['test']

    return ( 
        mnist_generator(test_data, test_label, test_batch_size, n_labelled, shuffle=0)
    )

