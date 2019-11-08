import os, sys
sys.path.append(os.getcwd())

import time
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf
from sklearn import svm
import cPickle as pickle

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot

parser = argparse.ArgumentParser()
parser.add_argument('--inputfile', type=str, default='Dateset_LRMnist_6.npz', help='npz file saves noisy mnist')
parser.add_argument('--checkpointdir', type=str, default='/media/alan/SteinsGate/github_code/mnist/TensorFlow_WGAN/improved_wgan_training-master/samples/Both/checkpoint', help='dir saves models')
parser.add_argument('--outputfile', type=str, default='./samples/deNoisedMnist.npz', help='npz file saves denoised mnist')
args = parser.parse_args()
#if not os.path.isdir(args.outdir):
#    os.makedirs(args.outdir)

MODE = 'wgan' # dcgan, wgan, or wgan-gp
DIM = 32 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
#ITERS = 300000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
Trained_model='./checkpoint_9919'
#Trained_model='/media/alan/SteinsGate/github_code/mnist/TensorFlow_WGAN/improved_wgan_training-master/samples/sigma1/POnly/classifier_checkpoint'

lib.print_model_settings(locals().copy())
#print('MSE + {} * adversarial'.format(args.weight))
'''
def save(saver, sess, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir,)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver.save(sess,os.path.join(checkpoint_dir, 'MNIST.model'), global_step=step)
'''

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


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise=None):
    _input = tf.reshape(noise, [-1, 1, 28, 28])
    output = lib.ops.conv2d.Conv2D('Generator_conv1.0',1, 2*DIM, 9, _input)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Generator_conv1.1',2*DIM, DIM, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Generator_conv1.2',DIM, 1, 5,output)
    #output = _input - output
    output = tf.clip_by_value(output,0,1)
    #output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])

def network(inputs,keep_prob):
	output = tf.reshape(inputs, [-1, 1, 28, 28])
        #output = tf.clip_by_value(output,0,1)
	#print(output.shape.as_list())
	output = lib.ops.conv2d.Conv2D('Conv1',1,10,5,output,Padding='VALID')
	#print(output.shape.as_list())
	output = tf.layers.max_pooling2d(output,2,2,data_format='channels_first') #https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling2d
	#print(output.shape.as_list())
	output = tf.nn.relu(output)
	output = lib.ops.conv2d.Conv2D('Conv2',10,20,5,output,Padding='VALID')
	output = tf.layers.max_pooling2d(output,2,2,data_format='channels_first')
	output = tf.nn.relu(output)
	#print(output.shape.as_list())
	output = tf.reshape(output, [-1, 320])
	output = lib.ops.linear.Linear('fc1', 320, 50, output)
	output = tf.nn.dropout(output, keep_prob) #https://blog.csdn.net/huahuazhu/article/details/73649389
	output = lib.ops.linear.Linear('fc2', 50, 10, output)
	output = tf.nn.softmax(output)
	return output

def compute_accuracy(v_xs,v_ys):
	global prediction
	y_pre = session.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
	correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	result = session.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
	return result
#placeholder for classifier
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32) 

#placeholder for denoise
input_noise= tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])

#build
prediction = network(xs,keep_prob)

fake_data = Generator(BATCH_SIZE,noise=input_noise)
disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

MSE_loss = tf.reduce_mean(tf.square(real_data - fake_data))
WDis_loss = -tf.reduce_mean(disc_fake) + tf.reduce_mean(disc_real)

'''logistic'''
#Logistic_model='./checkpoint_logistic'
Logistic_model='./checkpoint_logistic_woPooling'
'''
#avg_pooling
with tf.variable_scope('logistic'):
    W = tf.Variable(tf.zeros([196, 10]))
    b = tf.Variable(tf.zeros([10]))
    x_ = tf.reshape(xs, [-1, 1, 28, 28])
    x_ = tf.layers.average_pooling2d(x_,2,2,data_format='channels_first')
    x_ = tf.reshape(x_, [-1, 196])
    actv = tf.nn.softmax(tf.matmul(x_, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(actv), reduction_indices=1))
    pred = tf.equal(tf.argmax(actv, 1), tf.argmax(ys, 1))
    accr = tf.reduce_mean(tf.cast(pred, "float"))
'''
with tf.variable_scope('logistic'):
    W = tf.Variable(tf.zeros([784, 10]))
    #W = tf.Variable(tf.zeros([196, 10]))
    b = tf.Variable(tf.zeros([10]))
    #x_ = tf.reshape(x, [-1, 1, 28, 28])
    #x_ = tf.layers.average_pooling2d(x_,2,2,data_format='channels_first')
    #x_ = tf.reshape(x_, [-1, 196])
    actv = tf.nn.softmax(tf.matmul(xs, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(actv), reduction_indices=1))
    pred = tf.equal(tf.argmax(actv, 1), tf.argmax(ys, 1))
    accr = tf.reduce_mean(tf.cast(pred, "float"))


def generate_image(fixed_noise,flag=0):
    samples,MSE,WDis = session.run([fake_data,MSE_loss,WDis_loss],feed_dict={input_noise:fixed_noise[:,784:],real_data:fixed_noise[:,:784]})
    if flag==1:
        lib.save_images.save_images(
	    samples[:BATCH_SIZE,:].reshape((BATCH_SIZE, 28, 28)), 
	    args.outputfile.replace('.npz','_samples.png')
        )
        lib.save_images.save_images(
	    fixed_noise[:BATCH_SIZE,:784].reshape((BATCH_SIZE, 28, 28)), 
	    args.outputfile.replace('.npz','_original.png')
        )
        lib.save_images.save_images(
	    fixed_noise[:BATCH_SIZE,784:].reshape((BATCH_SIZE, 28, 28)), 
	    args.outputfile.replace('.npz','_Noised.png')
        )
    return samples,MSE,WDis

#saver
dict_0 = {}
dict_1 = {}
dict_2 = {}

for variable in tf.global_variables():
    key = variable.op.name
    if ('Generator_conv1' in key) or ('Discriminator' in key):
        dict_0[key] = variable
    elif 'logistic' in key:
	    dict_2[key] = variable
    else:
        dict_1[key] = variable

saver0 = tf.train.Saver(dict_0)
saver1 = tf.train.Saver(dict_1)
saver2 = tf.train.Saver(dict_2)
init = tf.initialize_all_variables()
session = tf.Session()
session.run(init) 

#load denioser
load(saver0, session, args.checkpointdir)
#load classifer
load(saver1, session, Trained_model)
#load logistic
load(saver2, session, Logistic_model)

#Test
# Dataset iterator
test_gen = lib.mnist.load_test(args.inputfile,BATCH_SIZE)
fixed_noise=[]
fixed_label=[]
denoised_image=[]
MSE=[]
WDis=[]
inital=1
for Timages,Tlabel in test_gen(shuffle=0):
    if inital==1:
        inital=0
        denoise_image,MSE_,WDis_=generate_image(Timages,flag=1)
        fixed_noise=Timages
        fixed_label=Tlabel
        denoised_image=denoise_image
    else:
        denoise_image,MSE_,WDis_=generate_image(Timages)
        fixed_noise=np.append(fixed_noise,Timages,axis=0)
        fixed_label = np.append(fixed_label,Tlabel,axis=0)
        denoised_image=np.append(denoised_image,denoise_image,axis=0)
    MSE.append(MSE_)
    WDis.append(WDis_)
    #print (np.array(MSE).shape,np.array(WDis).shape,fixed_noise.shape,denoised_image.shape,fixed_label.shape)
    

print('-------Test Rusult-------')
print('MSE: {} and Wasserstein distance: {}'.format(np.mean(MSE),np.mean(WDis)))
NetResult=compute_accuracy(fixed_noise[:,:784],fixed_label)
test_acc = session.run(accr, feed_dict={xs: fixed_noise[:,:784], ys: fixed_label})
print('Original image: Net:{}  SVM:{}'.format(NetResult,test_acc))
NetResult=compute_accuracy(fixed_noise[:,784:],fixed_label)
test_acc = session.run(accr, feed_dict={xs: fixed_noise[:,784:], ys: fixed_label})
print('Noisy image: Net:{}  SVM:{}'.format(NetResult,test_acc))
NetResult=compute_accuracy(denoised_image,fixed_label)
test_acc = session.run(accr, feed_dict={xs: denoised_image, ys: fixed_label})
print('Denoised image: Net:{}  SVM:{}'.format(NetResult,test_acc))
print('-------{}-------'.format(args.checkpointdir.split('/')[-2]))
print('{}  {}  {}  {}'.format(np.mean(MSE),np.mean(WDis),NetResult,test_acc))
print('---------------------')
print('---------------------')
print('---------------------')