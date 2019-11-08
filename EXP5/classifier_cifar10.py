import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os,scipy.misc
import cPickle
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.save_images
import random

#Load data
#Train/test data size:  50000/10000
image_size = 32
img_channels = 3

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict   


def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels


def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    #data = data.reshape([-1, img_channels, image_size, image_size])
    #data = data.transpose([0, 2, 3, 1])
    return data, labels


def prepare_data():
    print("======Loading data======")
    #download_data()
    data_dir = './Cifar10_data/cifar-10-batches-py'
    image_dim = image_size * image_size * img_channels
    meta = unpickle(data_dir + '/batches.meta')

    label_names = meta[b'label_names']
    label_count = len(label_names)
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels = load_data(train_files, data_dir, label_count)
    test_data, test_labels = load_data(['test_batch'], data_dir, label_count)

    train_data = train_data.astype('float32')
    train_data = train_data/255.0
    test_data = test_data.astype('float32')
    test_data = test_data/255.0

    print("Train data:", np.shape(train_data), np.shape(train_labels),np.dtype(train_data[0,0]))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    print("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[1] + 2*padding, oshape[2] + 2*padding)
    else:
        oshape = (oshape[1], oshape[2])
    new_batch = []
    npad = ((0, 0), (padding, padding), (padding, padding))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][:,nh:nh + crop_shape[0],
                                    nw:nw + crop_shape[1]]
    return np.array(new_batch)

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            #batch[i] = np.fliplr(batch[i])
            batch[i] = np.flipud(batch[i])
    return batch

def _random_brightness(batch, max_delta, seed=None):
    if max_delta < 0:
        raise ValueError('max_delta must be non-negative.')
    for i in range(len(batch)):
        delta = random.uniform(-max_delta, max_delta)
        batch[i] = batch[i] + delta
    return batch

def _random_contrast(image, lower, upper, seed=None):
    if upper <= lower:
        raise ValueError('upper must be > lower.')

    if lower < 0:
        raise ValueError('lower must be non-negative.')

  # Generate an a float in [lower, upper]
    for i in range(len(image)):
        contrast_factor = random.uniform(lower, upper)
        image[i] = (image[i]-np.mean(image[i]))*contrast_factor+np.mean(image[i])
    return image

def data_augmentation(batch):
    batch = batch.reshape([-1, 3, 32, 32])
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    #batch = _random_brightness(batch,max_delta=63.0/255.0)
    #batch = _random_contrast(batch,lower=0.2, upper=1.8)
    batch = batch.reshape([-1, 3*32*32])
    #lib.save_images.save_images(batch[:100,:].reshape((100, 3, 32, 32)),'samples.png')
    return batch


def network(inputs,keep_prob):
    '''
    #whiten
    mean, variance = tf.nn.moments(inputs, axes=1)
    mean=tf.expand_dims(mean, -1)
    variance=tf.expand_dims(variance, -1)
    inputs = (inputs-mean)/tf.maximum(tf.sqrt(variance),1.0/tf.sqrt(3.0*32.0*32.0))
    '''
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    #print(output.shape.as_list())
    output = lib.ops.conv2d.Conv2D('Conv1_1',3,64,3,output,Padding='SAME')
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Conv1_2',64,64,3,output,Padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output,2,2,data_format='channels_first') #https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling2d

    output = lib.ops.conv2d.Conv2D('Conv2_1',64,64,3,output,Padding='SAME')
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Conv2_2',64,64,3,output,Padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output,2,2,data_format='channels_first')

    output = lib.ops.conv2d.Conv2D('Conv3_1',64,64,3,output,Padding='SAME')
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Conv3_2',64,64,3,output,Padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output,2,2,data_format='channels_first')
    '''
    output = lib.ops.conv2d.Conv2D('Conv4_1',64,64,3,output,Padding='SAME')
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Conv4_2',64,64,3,output,Padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output,2,2,data_format='channels_first')
    '''#print(output.shape.as_list())'''
    output = tf.reshape(output, [-1, 4*4*64])
    output = lib.ops.linear.Linear('fc1', 4*4*64, 256, output)
    output = tf.nn.relu(output)
    output = tf.nn.dropout(output, keep_prob) #https://blog.csdn.net/huahuazhu/article/details/73649389
    output = lib.ops.linear.Linear('fc2', 256, 10, output)
    output = tf.nn.softmax(output)
    return output

'''
def add_layer(inputs,in_size,out_size,activation_function=None):
    biases = tf.Variable(tf.zeros([1,out_size])+0.03) 
    Weights = tf.Variable(tf.random_normal([in_size, out_size],mean=0,stddev=0.3))
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
'''

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

def save(saver, sess, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir,)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver.save(sess,os.path.join(checkpoint_dir, 'MNIST.model'), global_step=step)


def load(saver, sess, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

noise_std=0
Trained_model='./checkpoint'
#Trained_model=None #training

'''
#save noisy data
scale=4.
test_image=mnist.test.images.copy() #(10000, 784)
#print(test_image[0,:])
lib.save_images.save_images(test_image[-100:,:].reshape((100, 28, 28)),'Test_{}.png'.format('HR'))
test_image=np.reshape(test_image,(mnist.test.num_examples,28,28)) #(10000, 28, 28)
for i in range(mnist.test.num_examples):
    LR = scipy.misc.imresize(np.uint8(test_image[i,:,:]*255), 1./scale, 'bicubic','L')
    #print(LR.dtype)   === uint8
    ILR = scipy.misc.imresize(LR, scale, 'bicubic','L')
    test_image[i,:,:]=np.clip(np.float32(ILR)/255,0,1) 
 #(10000, 28, 28)
test_image=np.reshape(test_image,(mnist.test.num_examples,28*28))#(10000, 784)
lib.save_images.save_images(test_image[-100:,:].reshape((100, 28, 28)),'Test_{}.png'.format('ILR'))
lib.save_images.save_images(mnist.test.images[-100:,:].reshape((100, 28, 28)),'Test_{}.png'.format('HR2'))
#print(test_image[0,:])

valid_image=mnist.validation.images.copy()
lib.save_images.save_images(valid_image[-100:,:].reshape((100, 28, 28)),'Valid_{}.png'.format('HR'))
valid_image=np.reshape(valid_image,(mnist.validation.num_examples,28,28)) 
for i in range(mnist.validation.num_examples):
    LR = scipy.misc.imresize(np.uint8(valid_image[i,:,:]*255), 1./scale, 'bicubic','L')
    ILR = scipy.misc.imresize(LR, scale, 'bicubic','L')
    valid_image[i,:,:]=np.clip(np.float32(ILR)/255,0,1) 
valid_image=np.reshape(valid_image,(mnist.validation.num_examples,28*28))
lib.save_images.save_images(valid_image[-100:,:].reshape((100, 28, 28)),'Valid_{}.png'.format('ILR'))

train_image=mnist.train.images.copy()
lib.save_images.save_images(train_image[-100:,:].reshape((100, 28, 28)),'Train_{}.png'.format('HR'))
train_image=np.reshape(train_image,(mnist.train.num_examples,28,28)) 
for i in range(mnist.train.num_examples):
    LR = scipy.misc.imresize(np.uint8(train_image[i,:,:]*255), 1./scale, 'bicubic','L')
    ILR = scipy.misc.imresize(LR, scale, 'bicubic','L')
    train_image[i,:,:]=np.clip(np.float32(ILR)/255,0,1) 
train_image=np.reshape(train_image,(mnist.train.num_examples,28*28))
lib.save_images.save_images(train_image[-100:,:].reshape((100, 28, 28)),'Train_{}.png'.format('ILR'))
#np.savez("NoisyMnist_{}.npz".format(noise_std), test=test_image, valid=valid_image, train = train_image)
#r = np.load("NoisyMnist_1.npz")
#valid=r['valid']
test_dataset=np.hstack([mnist.test.images,test_image])
valid_dataset=np.hstack([mnist.validation.images,valid_image])
train_dataset=np.hstack([mnist.train.images,train_image])
np.savez("Dateset_LRMnist_{}.npz".format(np.int(scale)), test=test_dataset, valid=valid_dataset, train = train_dataset)
np.savez("Dateset_Label_{}.npz".format(np.int(scale)), test=mnist.test.labels, valid=mnist.validation.labels, train = mnist.train.labels)
exit()
'''

xs = tf.placeholder(tf.float32,[None,3072])
ys = tf.placeholder(tf.float32,[None,10])
learning_rate=tf.placeholder(tf.float32, shape=[], name='learning_rate')
keep_prob = tf.placeholder(tf.float32)  
 
prediction = network(xs,keep_prob)
prediction=tf.clip_by_value(prediction,1e-15,1.0)
 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
 
optimizer = tf.train.GradientDescentOptimizer(learning_rate)  

gvs = optimizer.compute_gradients(cross_entropy)
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
train_step = optimizer.apply_gradients(capped_gvs)  # clip gradients
'''
train_step =optimizer.minimize(cross_entropy)
'''
saver = tf.train.Saver()
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init) 
#load and test
if not Trained_model==None:
    load(saver, sess, Trained_model)
    _, _, test_x, test_y = prepare_data()
    test_image=test_x
    print(test_image.dtype)
    if not noise_std==0:
        #noise=tf.random_normal(test_image.shape, stddev=noise_std)
        noise=np.random.normal(0,noise_std,test_image.shape)
        print(noise.dtype)
        test_image=test_image+noise
    #save sample
    lib.save_images.save_images(test_image[:100,:].reshape((100, 3, 32, 32)),'samples_{}.png'.format(noise_std))
    result=compute_accuracy(test_image,test_y)
    print(result)
    exit()

#train
Initial_lr=0.1
batch_size=100
iterration=50000/batch_size
train_x, train_y, test_x, test_y = prepare_data()
Best_result=0.83
for i in range(iterration*100000):
    pre_index= (i%iterration)*batch_size
    batch_xs = data_augmentation(train_x[pre_index:pre_index+batch_size])
    batch_ys = train_y[pre_index:pre_index+batch_size]
    sess.run(train_step,feed_dict = {xs:batch_xs,ys:batch_ys,keep_prob:0.5,learning_rate:Initial_lr})
    #print(compute_accuracy(batch_xs,batch_ys))
    if i%iterration == 0:
        result=compute_accuracy(test_x,test_y)
        print(i//iterration,result,compute_accuracy(batch_xs,batch_ys)) #https://blog.csdn.net/mzpmzk/article/details/78647730
        if i//iterration==100:
            Initial_lr=Initial_lr*0.1
            print('lr='+str(Initial_lr))
        if i//iterration==200:
            Initial_lr=Initial_lr*0.2
            print('lr='+str(Initial_lr))
        if i//iterration==500:
            Initial_lr=Initial_lr*0.5
            print('lr='+str(Initial_lr))
        if i//iterration==1500:
            Initial_lr=Initial_lr*0.1
            print('lr='+str(Initial_lr))
        if result>Best_result:
            print('Best accuracy: {} and save as {}'.format(result,i))
            Best_result=result
            save(saver,sess,'./checkpoint',i//iterration)


'''
#--------------network1 Tensorflow (w/o suffix)-----------------
#https://www.jianshu.com/p/4ed7f7b15736
def network(inputs,keep_prob):
    #whiten
    mean, variance = tf.nn.moments(inputs, axes=1)
    mean=tf.expand_dims(mean, -1)
    variance=tf.expand_dims(variance, -1)
    inputs = (inputs-mean)/tf.maximum(tf.sqrt(variance),1.0/tf.sqrt(3.0*32.0*32.0))
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    #output = tf.image.per_image_standardization(output)
    #print(output.shape.as_list())
    output = lib.ops.conv2d.Conv2D('Conv1',3,64,5,output,Padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output,3,2,padding='SAME',data_format='channels_first') #https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling2d
    output = tf.nn.lrn(output, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
    output = lib.ops.conv2d.Conv2D('Conv2',64,64,5,output,Padding='SAME')
    output = tf.nn.relu(output)
    output = tf.nn.lrn(output, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    output = tf.layers.max_pooling2d(output,3,2,padding='SAME',data_format='channels_first')

    #print(output.shape.as_list())
    output = tf.reshape(output, [-1, 8*8*64])
    output = lib.ops.linear.Linear('fc1', 8*8*64, 384, output)
    output = tf.nn.relu(output)
    output1 = tf.nn.dropout(output, keep_prob) #https://blog.csdn.net/huahuazhu/article/details/73649389

    output = lib.ops.linear.Linear('fc2', 384, 192, output)
    output = tf.nn.relu(output)

    output = lib.ops.linear.Linear('fc3', 192, 10, output)
    output = tf.nn.softmax(output)
    return output

#--------------network2 simplified parameter 1/5 (ver22)-----------------
def network(inputs,keep_prob):
    #whiten
    mean, variance = tf.nn.moments(inputs, axes=1)
    mean=tf.expand_dims(mean, -1)
    variance=tf.expand_dims(variance, -1)
    inputs = (inputs-mean)/tf.maximum(tf.sqrt(variance),1.0/tf.sqrt(3.0*32.0*32.0))
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    #print(output.shape.as_list())
    output = lib.ops.conv2d.Conv2D('Conv1_1',3,64,3,output,Padding='SAME')
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Conv1_2',64,64,3,output,Padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output,2,2,data_format='channels_first') #https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling2d

    output = lib.ops.conv2d.Conv2D('Conv2_1',64,64,3,output,Padding='SAME')
    output = tf.nn.relu(output)
    #output = lib.ops.conv2d.Conv2D('Conv2_2',64,64,3,output,Padding='SAME')
    #output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output,2,4,data_format='channels_first')

    output = lib.ops.conv2d.Conv2D('Conv3_1',64,64,3,output,Padding='SAME')
    output = tf.nn.relu(output)
    #output = lib.ops.conv2d.Conv2D('Conv3_2',64,64,3,output,Padding='SAME')
    #output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output,2,2,data_format='channels_first')

    #output = lib.ops.conv2d.Conv2D('Conv4_1',64,64,3,output,Padding='SAME')
    #output = tf.nn.relu(output)
    #output = lib.ops.conv2d.Conv2D('Conv4_2',64,64,3,output,Padding='SAME')
    #output = tf.nn.relu(output)
    #output = tf.layers.max_pooling2d(output,2,2,data_format='channels_first')
    #print(output.shape.as_list())
    output = tf.reshape(output, [-1, 4*4*64])
    output = lib.ops.linear.Linear('fc1', 4*4*64, 256, output)
    output = tf.nn.relu(output)
    output = tf.nn.dropout(output, keep_prob) #https://blog.csdn.net/huahuazhu/article/details/73649389
    output = lib.ops.linear.Linear('fc2', 256, 10, output)
    output = tf.nn.softmax(output)
    return output

#--------------network3 Equal (ver3)-----------------
#https://blog.csdn.net/laurenitum0716/article/details/79313749
def network(inputs,keep_prob):
    #whiten
    mean, variance = tf.nn.moments(inputs, axes=1)
    mean=tf.expand_dims(mean, -1)
    variance=tf.expand_dims(variance, -1)
    inputs = (inputs-mean)/tf.maximum(tf.sqrt(variance),1.0/tf.sqrt(3.0*32.0*32.0))

    output = tf.reshape(inputs, [-1, 3, 32, 32])
    #print(output.shape.as_list())
    output = lib.ops.conv2d.Conv2D('Conv1',3,64,3,output,Padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output,2,2,data_format='channels_first') #https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling2d

    output = lib.ops.conv2d.Conv2D('Conv2',64,128,3,output,Padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output,2,2,data_format='channels_first')

    output = lib.ops.conv2d.Conv2D('Conv3',128,256,3,output,Padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output,2,2,data_format='channels_first')

    output = lib.ops.conv2d.Conv2D('Conv4',256,512,3,output,Padding='SAME')
    output = tf.nn.relu(output)
    output = tf.layers.max_pooling2d(output,2,2,data_format='channels_first')
    #print(output.shape.as_list())
    output = tf.reshape(output, [-1, 2*2*512])
    output = lib.ops.linear.Linear('fc1', 2*2*512, 256, output)
    output = tf.nn.relu(output)
    output = tf.nn.dropout(output, keep_prob) #https://blog.csdn.net/huahuazhu/article/details/73649389
    output = lib.ops.linear.Linear('fc2', 256, 10, output)
    output = tf.nn.softmax(output)
    return output
'''