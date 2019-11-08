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
import tflib.cifar10
import tflib.plot

parser = argparse.ArgumentParser()
parser.add_argument('--inputfile', type=str, default='Dateset_LRMnist_3.npz', help='npz file saves noisy mnist')
parser.add_argument('--checkpointdir', type=str, default='/media/alan/SteinsGate/github_code/mnist/TensorFlow_WGAN/Cifar10/samples/CPD00/checkpoint', help='dir saves models')
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
OUTPUT_DIM = 3072 # Number of pixels in MNIST (28*28)
Trained_model='./checkpoint8518'
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
    _input = tf.reshape(noise, [-1, 3, 32, 32])
    output = lib.ops.conv2d.Conv2D('Generator_conv1.0',3, 2*DIM, 9, _input)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Generator_conv1.1',2*DIM, DIM, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Generator_conv1.2',DIM, 3, 5,output)
    #output = _input - output
    output = tf.clip_by_value(output,0,1)
    #output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])


def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Discriminator.1',3,DIM,5,output,stride=2)
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

    output = tf.reshape(output, [-1, 4*4*64])
    output = lib.ops.linear.Linear('fc1', 4*4*64, 256, output)
    output = tf.nn.relu(output)
    output = tf.nn.dropout(output, keep_prob) #https://blog.csdn.net/huahuazhu/article/details/73649389
    output = lib.ops.linear.Linear('fc2', 256, 10, output)
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
xs = tf.placeholder(tf.float32,[None,OUTPUT_DIM])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32) 

#placeholder for denoise
input_noise= tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])

#build
prediction = network(xs,keep_prob)
prediction=tf.clip_by_value(prediction,1e-15,1.0)

fake_data = Generator(BATCH_SIZE,noise=input_noise)
disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

MSE_loss = tf.reduce_mean(tf.square(real_data - fake_data))
WDis_loss = -tf.reduce_mean(disc_fake) + tf.reduce_mean(disc_real)

def generate_image(fixed_noise,flag=0):
    samples,MSE,WDis = session.run([fake_data,MSE_loss,WDis_loss],feed_dict={input_noise:fixed_noise[:,OUTPUT_DIM:],real_data:fixed_noise[:,:OUTPUT_DIM]})
    if flag==1:
        lib.save_images.save_images(
	    samples[:BATCH_SIZE,:].reshape((BATCH_SIZE, 3, 32, 32)), 
	    args.outputfile.replace('.npz','_samples.png')
        )
        lib.save_images.save_images(
	    fixed_noise[:BATCH_SIZE,:OUTPUT_DIM].reshape((BATCH_SIZE, 3, 32, 32)), 
	    args.outputfile.replace('.npz','_original.png')
        )
        lib.save_images.save_images(
	    fixed_noise[:BATCH_SIZE,OUTPUT_DIM:].reshape((BATCH_SIZE, 3, 32, 32)), 
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
#saver2 = tf.train.Saver(dict_2)
init = tf.initialize_all_variables()
session = tf.Session()
session.run(init) 

#load denioser
load(saver0, session, args.checkpointdir)
#load classifer
load(saver1, session, Trained_model)
#load logistic
#load(saver2, session, Logistic_model)

#Test
# Dataset iterator
_, _, test_gen = lib.cifar10.load(args.inputfile, 10, 10, BATCH_SIZE)
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
NetResult=compute_accuracy(fixed_noise[:,:OUTPUT_DIM],fixed_label)
#test_acc = session.run(accr, feed_dict={xs: fixed_noise[:,:OUTPUT_DIM], ys: fixed_label})
print('Original image: Net:{}'.format(NetResult))
NetResult=compute_accuracy(fixed_noise[:,OUTPUT_DIM:],fixed_label)
print('Noisy image: Net:{}'.format(NetResult))
NetResult=compute_accuracy(denoised_image,fixed_label)
print('Denoised image: Net:{}'.format(NetResult))
print('-------{}-------'.format(args.checkpointdir.split('/')[-2]))
print('{}  {}  {}'.format(np.mean(MSE),np.mean(WDis),NetResult))
print('---------------------')
print('---------------------')
print('---------------------')