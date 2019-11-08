#28*28 + padding = 30*30  downsample 6
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os,scipy.misc,cv2

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.save_images
 
mnist = input_data.read_data_sets('MNIST_data',one_hot=True) #Train/validation/test data size:  55000/5000/10000

def network(inputs,keep_prob):
    output = tf.reshape(inputs, [-1, 1, 28, 28])
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

inputfile='LRMnist_6.0.npz' # 'NoisyMnist_1.npz' "deNoisedMnist.npz"
Trained_model='./checkpoint_9919'

'''
#save LR data
scale=6.
test_image=mnist.test.images #(10000, 784)
#print(test_image[0,:])
lib.save_images.save_images(test_image[-100:,:].reshape((100, 28, 28)),'Test_{}.png'.format('HR'))
test_image=np.reshape(test_image,(mnist.test.num_examples,28,28)) #(10000, 28, 28)
for i in range(mnist.test.num_examples):
    HR = cv2.copyMakeBorder(np.uint8(test_image[i,:,:]*255),1,1,1,1,cv2.BORDER_REPLICATE)
    LR = scipy.misc.imresize(HR, 1./scale, 'bicubic','L')
    #print(LR.dtype)   === uint8
    ILR = scipy.misc.imresize(LR, scale, 'bicubic','L')
    ILR=ILR[1:-1,1:-1]
    test_image[i,:,:]=np.clip(np.float32(ILR)/255,0,1) 
 #(10000, 28, 28)
test_image=np.reshape(test_image,(mnist.test.num_examples,28*28))#(10000, 784)
lib.save_images.save_images(test_image[-100:,:].reshape((100, 28, 28)),'Test_{}.png'.format('ILR'))
#print(test_image[0,:])

valid_image=mnist.validation.images
lib.save_images.save_images(valid_image[-100:,:].reshape((100, 28, 28)),'Valid_{}.png'.format('HR'))
valid_image=np.reshape(valid_image,(mnist.validation.num_examples,28,28)) 
for i in range(mnist.validation.num_examples):
    HR = cv2.copyMakeBorder(np.uint8(valid_image[i,:,:]*255),1,1,1,1,cv2.BORDER_REPLICATE)
    LR = scipy.misc.imresize(HR, 1./scale, 'bicubic','L')
    ILR = scipy.misc.imresize(LR, scale, 'bicubic','L')
    ILR=ILR[1:-1,1:-1]
    valid_image[i,:,:]=np.clip(np.float32(ILR)/255,0,1) 
valid_image=np.reshape(valid_image,(mnist.validation.num_examples,28*28))
lib.save_images.save_images(valid_image[-100:,:].reshape((100, 28, 28)),'Valid_{}.png'.format('ILR'))

train_image=mnist.train.images
lib.save_images.save_images(train_image[-100:,:].reshape((100, 28, 28)),'Train_{}.png'.format('HR'))
train_image=np.reshape(train_image,(mnist.train.num_examples,28,28)) 
for i in range(mnist.train.num_examples):
    HR = cv2.copyMakeBorder(np.uint8(train_image[i,:,:]*255),1,1,1,1,cv2.BORDER_REPLICATE)
    LR = scipy.misc.imresize(HR, 1./scale, 'bicubic','L')
    ILR = scipy.misc.imresize(LR, scale, 'bicubic','L')
    ILR=ILR[1:-1,1:-1]
    train_image[i,:,:]=np.clip(np.float32(ILR)/255,0,1) 
train_image=np.reshape(train_image,(mnist.train.num_examples,28*28))
lib.save_images.save_images(train_image[-100:,:].reshape((100, 28, 28)),'Train_{}.png'.format('ILR'))

np.savez("LRMnist_{}.npz".format(scale), test=test_image, valid=valid_image, train = train_image)
exit()
#r = np.load("NoisyMnist_1.npz")
#train_image=r['train']
'''

xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)  
 
prediction = network(xs,keep_prob)
 
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
 
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  

saver = tf.train.Saver()
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init) 
#load and test
if not Trained_model==None:
    load(saver, sess, Trained_model)
    r = np.load(inputfile)    
    #test_image=mnist.test.images
    test_image=r['test']    
    if 'label' in r.keys():
        print('load label from npz')
        test_label=r['label']
    else:
        print('load label from raw')
        test_label=mnist.test.labels
        
    '''
    test_image=mnist.test.images
    print(test_image.dtype)
    if not noise_std==0:
        #noise=tf.random_normal(test_image.shape, stddev=noise_std)
        noise=np.random.normal(0,noise_std,test_image.shape)
        print(noise.dtype)
        test_image=test_image+noise
    #save sample
    lib.save_images.save_images(test_image[:100,:].reshape((100, 28, 28)),'samples_{}.png'.format(noise_std))
    '''
    result=compute_accuracy(test_image,test_label)
    print(result)
    #exit()
'''
#train
Best_result=0.99
for i in range(550*140):
    batch_xs,batch_ys = mnist.train.next_batch(100) 
    sess.run(train_step,feed_dict = {xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i%550 == 0:
        result=compute_accuracy(mnist.test.images,mnist.test.labels)
        print(i//550,result) #https://blog.csdn.net/mzpmzk/article/details/78647730
        if result>Best_result:
            print('Best accuracy: {} and save as {}'.format(result,i))
            Best_result=result
            save(saver,sess,'./checkpoint',i//550)
'''
