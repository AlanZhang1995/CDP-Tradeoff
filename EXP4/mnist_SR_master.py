#SRCNN structure  + Add noise on line + select best G_loss model + MSE decay + Tensorboard + weight3*0.1
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

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot

parser = argparse.ArgumentParser()
#parser.add_argument('--inputfile', type=str, default='Dateset_NoisyMnist_1.npz', help='npz file saves noisy mnist')
parser.add_argument('--sigma', type=int, default=6, help='Downsample factor')
parser.add_argument('--weight1', type=float, default=1, help='loss weight for MSE')
parser.add_argument('--weight2', type=float, default=0, help='loss weight for GAN')
parser.add_argument('--weight3', type=float, default=0, help='loss weight for Classifer')
parser.add_argument('--pretrained', type=str, default=None, help='checkpoint dir for pretrained denoiser')
parser.add_argument('--outdir', type=str, default='/media/alan/SteinsGate/github_code/mnist/TensorFlow_WGAN/improved_wgan_training-sr/samples/mnist', help='Dir for saving samples')
args = parser.parse_args()
if not os.path.isdir(args.outdir):
    os.makedirs(args.outdir)
    os.makedirs(args.outdir+'/logs/')
inputfile='Dateset_LRMnist_{}'.format(args.sigma)+'.npz'

#some settings
MODE = 'wgan' # dcgan, wgan, or wgan-gp
DIM = 32 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 40000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
Learning_Rate=1e-3*5
Trained_model='./checkpoint_9919'
lib.print_model_settings(locals().copy())
print('Downsample factor is {}'.format(args.sigma))
print('loss= {} * MSE + {} * adversarial + {} * classifer'.format(args.weight1,args.weight2,args.weight3))


#----------------------------network------------------------------------
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
    output = lib.ops.conv2d.Conv2D('Conv1',1,10,5,output,Padding='VALID')
    output = tf.layers.max_pooling2d(output,2,2,data_format='channels_first') #https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling2d
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Conv2',10,20,5,output,Padding='VALID')
    output = tf.layers.max_pooling2d(output,2,2,data_format='channels_first')
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 320])
    output = lib.ops.linear.Linear('fc1', 320, 50, output)
    output = tf.nn.dropout(output, keep_prob) #https://blog.csdn.net/huahuazhu/article/details/73649389
    output = lib.ops.linear.Linear('fc2', 50, 10, output)
    output = tf.nn.softmax(output)
    return output
#-----------------------------------------------------------------------

#functions used for saving and loading models
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


#functions with other functions
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = session.run(Tprediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = session.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result


def generate_image(frame, true_dist, input_dist, flag=0):
    if flag==0:
        #samples,TMSE,TWDis = session.run([fixed_noise_samples,MSE,WDis],feed_dict={Tinput_noise:fixed_noise[:,784:],Treal_data:fixed_noise[:,:784]})
        samples = session.run(fixed_noise_samples,feed_dict={Tinput_noise:fixed_noise[:,784:]})
        #samples = np.clip(samples,0,1)
        print('-------Test Rusult: iter {}-------'.format(frame))
            #print('MSE: {} and Wasserstein distance: {}'.format(TMSE,TWDis))
        lib.save_images.save_images(
        samples[:100,:].reshape((100, 28, 28)), 
        args.outdir+'/samples_{}.png'.format(frame)
        )
        result=compute_accuracy(samples,fixed_label)
        print('Denoised Accuracy: {}'.format(result))
        lib.plot.plot(args.outdir+'/test accuracy', result)
        print('-------------------------------------')

    else:
        lib.save_images.save_images(
                input_dist.reshape((BATCH_SIZE, 28, 28)), 
                args.outdir+'/input_{}.png'.format(frame)
            )
        lib.save_images.save_images(
                true_dist.reshape((BATCH_SIZE, 28, 28)), 
                args.outdir+'/real_{}.png'.format(frame)
            )
        lib.save_images.save_images(
                fixed_noise[:100,:784].reshape((100, 28, 28)), 
                args.outdir+'/test_real_{}.png'.format(frame)
            )
        lib.save_images.save_images(
                fixed_noise[:100,784:].reshape((100, 28, 28)), 
                args.outdir+'/test_noise_{}.png'.format(frame)
            )
        result=0
    return result


#---------------------------------build net---------------------------------
""" Build Graph """
#train--input
input_noise= tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
learning_rate=tf.placeholder(tf.float32, shape=[], name='learning_rate_for_G')
MSE_weight=tf.placeholder(tf.float32, shape=[], name='MSE_weight')

ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)  

#train--build
fake_data = Generator(BATCH_SIZE,noise=input_noise)
disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)
prediction = network(fake_data,keep_prob)
#####prediction_real = network(real_data,keep_prob)

#test--input
Tinput_noise= tf.placeholder(tf.float32, shape=[10000, OUTPUT_DIM])
#Treal_data = tf.placeholder(tf.float32, shape=[10000, OUTPUT_DIM])
xs = tf.placeholder(tf.float32,[None,784])
#test--build
fixed_noise_samples = Generator(10000, noise=Tinput_noise)
Tprediction = network(xs,keep_prob)


gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'wgan':
    """ Loss Function """
    #for G
    MSE = tf.reduce_mean(tf.square(real_data - fake_data))
    WDis = -tf.reduce_mean(disc_fake)
    cross_entropy1 = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
    #####cross_entropy2 =tf.reduce_mean(-tf.reduce_sum(prediction_real*tf.log(prediction),reduction_indices=[1]))
    cross_entropy=cross_entropy1
    #####cross_entropy = cross_entropy1  + cross_entropy2 * 0
    gen_cost = MSE * MSE_weight + WDis * args.weight2 + cross_entropy * args.weight3
    #for select G
    WDis2 = -tf.reduce_mean(disc_fake) + tf.reduce_mean(disc_real)
    gen_cost2 = MSE * MSE_weight + WDis2 * args.weight2 + cross_entropy * args.weight3
    #for D
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    tf.summary.scalar('MSE', MSE)
    tf.summary.scalar('EM distance', WDis)
    tf.summary.scalar('Cross Entropy', cross_entropy)
    tf.summary.scalar('G loss', gen_cost)
    tf.summary.scalar('G loss2', gen_cost2)
    tf.summary.scalar('D loss', disc_cost)
    tf.summary.scalar('MSE weight', MSE_weight)
    tf.summary.scalar('Learning rate', learning_rate)
    
    """ optimizers """
    '''
    gen_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(disc_cost, var_list=disc_params)
    '''
    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=learning_rate, 
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)
    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake, 
        tf.ones_like(disc_fake)
    ))

    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake, 
        tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_real, 
        tf.ones_like(disc_real)
    ))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params) #tf.train.AdamOptimizer
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None
#-----------------------------------------------------------------------

#-------------------------------T/V/T data------------------------------
# Dataset iterator
train_gen, dev_gen, _ = lib.mnist.load(inputfile, BATCH_SIZE, BATCH_SIZE)
test_gen = lib.mnist.load_test(inputfile,10000)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images,targets
    '''
    for images,targets in dev_gen():
            yield images
    for images,targets in test_gen():
            yield images
    '''

# For saving samples
for Timages,Tlabel in test_gen():
    fixed_noise = Timages
    fixed_label = Tlabel
    break


#---------------------------------Training---------------------------------
""" saver """
#saver
dict_0 = {}
dict_1 = {}

for variable in tf.global_variables():
    key = variable.op.name
    if ('Generator_conv1' in key) or ('Discriminator' in key):
        dict_0[key] = variable
    elif ('Conv' in key) or ('fc' in key):
        dict_1[key] = variable

saver0 = tf.train.Saver(dict_0,max_to_keep=2)
saver1 = tf.train.Saver(dict_1,max_to_keep=1)

"""Train loop"""
Best_GLoss=1000
Best_Accuracy=0
# Train loop
with tf.Session() as session:
    #tensorboard
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.outdir+"/logs/train", session.graph)
    valid_writer = tf.summary.FileWriter(args.outdir+"/logs/valid", session.graph)

    #initialization
    session.run(tf.initialize_all_variables())
    
    #load classifer
    load(saver1, session, Trained_model)
    #load pretrained denoiser
    if not args.pretrained==None:
        load(saver0, session, args.pretrained)

    #training data
    gen = inf_train_gen()

    for iteration in xrange(ITERS):
        #start_time = time.time()
        #MSE weight decay
        MSE_weight_= max(args.weight1, 1. - iteration/299.)

        #Learning Rate Decay
        if iteration% 10000 == 0 :
            Learning_Rate=Learning_Rate/5
            print('Iteration {}: change Learning Rate to {}'.format(iteration,Learning_Rate))
        #Update G
        if iteration > 0:
            Dis_val,MSE_val,entropy_val,_,summary = session.run([WDis,MSE,cross_entropy,gen_train_op,merged],feed_dict={real_data: _data[:,:784],input_noise:_data[:,784:],ys:_label,learning_rate:Learning_Rate,keep_prob:1,MSE_weight:MSE_weight_})

        #Update D
        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data,_label = gen.next()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data[:,:784],input_noise:_data[:,784:]}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        #log--train
        if (iteration > 0 and iteration < 5) or (iteration % 100 == 99):
            lib.plot.plot(args.outdir+'/train MSE', MSE_val)
            lib.plot.plot(args.outdir+'/train Wasserstein distance', Dis_val)
            lib.plot.plot(args.outdir+'/train cross entropy', entropy_val)
            lib.plot.plot(args.outdir+'/train disc cost', _disc_cost)
            #lib.plot.plot(args.outdir+'/time', time.time() - start_time)
            train_writer.add_summary(summary, iteration)

        """Validation"""
        # Calculate dev loss and generate samples every 1000 iters
        if (iteration == 0) or (iteration % 100 == 99 and iteration<40000) or (iteration % 1000 == 999 and iteration>40000):
            dev_disc_costs = []
            dev_MSE = []
            dev_Dis = []
            dev_entropy = []
            dev_gen_cost = []
            for images,labels in dev_gen(): #for images,_ in test_gen():
                _dev_gen_cost,_dev_disc_cost,_dev_MSE,_dev_Dis,_dev_entropy, _dev_summary= session.run(
                    [gen_cost2,disc_cost, MSE,WDis,cross_entropy,merged],
                    feed_dict={real_data: images[:,:784],input_noise:images[:,784:],ys:labels,learning_rate:Learning_Rate,keep_prob:1,MSE_weight:MSE_weight_}
                )
                dev_disc_costs.append(_dev_disc_cost)
                dev_MSE.append(_dev_MSE)
                dev_Dis.append(_dev_Dis)
                dev_entropy.append(_dev_entropy)
                dev_gen_cost.append(_dev_gen_cost)
            #log--valid
            lib.plot.plot(args.outdir+'/dev disc cost', np.mean(dev_disc_costs))
            lib.plot.plot(args.outdir+'/dev MSE', np.mean(dev_MSE))
            lib.plot.plot(args.outdir+'/dev Wasserstein distance', np.mean(dev_Dis))
            lib.plot.plot(args.outdir+'/dev cross entropy', np.mean(dev_entropy))
            valid_writer.add_summary(_dev_summary, iteration)

            GLoss=np.mean(dev_gen_cost)
            #test classification accuracy
            if (iteration % 100 == 99 and iteration<5000) or (iteration % 1000 == 999 and iteration>5000):
                Accuracy=generate_image(iteration, _data[:,:784], _data[:,784:])
                if Best_Accuracy<Accuracy:
                    Best_Accuracy=Accuracy
        #save models
        if (iteration > 25000) and (GLoss < Best_GLoss):
            Best_GLoss=GLoss
            print('-----------------------------------------------------------')
            print('save best GLoss model: iter {}  GLoss: {} Best_accu: {}'.format(iteration, Best_GLoss, Best_Accuracy))
            Accuracy_=generate_image(iteration, _data[:,:784], _data[:,784:])
            print('-----------------------------------------------------------')
            save(saver0,session,args.outdir+'/checkpoint',iteration)
            #save(saver1,session,args.outdir+'/classifier_checkpoint',iteration)

        if (iteration% 20 == 0) and (iteration<220):
            generate_image(iteration, _data[:,:784], _data[:,784:],flag=1)
    
        # display logs
        if (iteration < 5) or (iteration % 100 == 99 and iteration<40000) or (iteration % 1000 == 999 and iteration>40000):
            lib.plot.flush()
    
        lib.plot.tick()
