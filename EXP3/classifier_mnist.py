import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.save_images
 
mnist = input_data.read_data_sets('MNIST_data',one_hot=True) #Train/validation/test data size:  55000/5000/10000

def network(inputs,keep_prob):
    output = tf.reshape(inputs, [-1, 1, 28, 28])
    #print(output.shape.as_list())
    output = lib.ops.conv2d.Conv2D('Conv1_1',1,64,3,output,Padding='SAME')
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

    output = tf.reshape(output, [-1, 3*3*64])
    output = lib.ops.linear.Linear('fc1', 3*3*64, 256, output)
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
Trained_model='./checkpoint_net2_9938'
#Trained_model=None #training

'''
#save noisy data
noise=np.random.normal(0,noise_std,mnist.test.images.shape)
test_image=mnist.test.images+noise
noise=np.random.normal(0,noise_std,mnist.validation.images.shape)
valid_image=mnist.validation.images+noise
noise=np.random.normal(0,noise_std,mnist.train.images.shape)
train_image=mnist.train.images+noise
np.savez("NoisyMnist_{}.npz".format(noise_std), test=test_image, valid=valid_image, train = train_image)
#r = np.load("NoisyMnist_1.npz")
#valid=r['valid']
test_dataset=np.hstack([mnist.test.images,test_image])
valid_dataset=np.hstack([mnist.validation.images,valid_image])
train_dataset=np.hstack([mnist.train.images,train_image])
np.savez("Dateset_NoisyMnist_{}.npz".format(noise_std), test=test_dataset, valid=valid_dataset, train = train_dataset)
np.savez("Dateset_Label_{}.npz".format(noise_std), test=mnist.test.labels, valid=mnist.validation.labels, train = mnist.train.labels)
'''
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)  
learning_rate=tf.placeholder(tf.float32, shape=[])
 
prediction = network(xs,keep_prob)
prediction=tf.clip_by_value(prediction,1e-15,10000000000000000)
 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
 
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)  

saver = tf.train.Saver(max_to_keep=1)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init) 
#load and test
if not Trained_model==None:
    load(saver, sess, Trained_model)
    test_image=mnist.test.images
    print(test_image.dtype)
    if not noise_std==0:
        #noise=tf.random_normal(test_image.shape, stddev=noise_std)
        noise=np.random.normal(0,noise_std,test_image.shape)
        print(noise.dtype)
        test_image=test_image+noise
    #save sample
    lib.save_images.save_images(test_image[:100,:].reshape((100, 28, 28)),'samples_{}.png'.format(noise_std))
    result=compute_accuracy(test_image,mnist.test.labels)
    print(result)
    exit()

#train
lr=0.01
print('lr:{}'.format(lr))
Best_result=0.99
for i in range(550*1000):
    batch_xs,batch_ys = mnist.train.next_batch(100) 
    sess.run(train_step,feed_dict = {xs:batch_xs,ys:batch_ys,keep_prob:0.5,learning_rate:lr})
    if i%550 == 0:
        result=compute_accuracy(mnist.test.images,mnist.test.labels)
        print(i//550,result) #https://blog.csdn.net/mzpmzk/article/details/78647730
        if result>Best_result:
            print('Best accuracy: {} and save as {}'.format(result,i))
            Best_result=result
            save(saver,sess,'./checkpoint',i//550)
        if i//550 == 200:
        	lr=lr*0.1
        	print('lr:{}'.format(lr))
