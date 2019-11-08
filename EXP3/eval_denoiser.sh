#!/bin/bash
python mnist_eval_master11.py --checkpointdir /media/alan/SteinsGate/github_code/mnist/TensorFlow_WGAN/improved_wgan_training-master-sort/samples/$1/checkpoint
#python mnist_eval_master2.py --inputfile Dateset_NoisyMnist_1.npz --checkpointdir /media/alan/SteinsGate/github_code/mnist/TensorFlow_WGAN/improved_wgan_training-master/samples/sigma1_save/sigma1_BN1/$1/checkpoint

mv /media/alan/SteinsGate/github_code/mnist/TensorFlow_WGAN/improved_wgan_training-master-sort/samples/deNoisedMnist_samples.png /media/alan/SteinsGate/github_code/mnist/TensorFlow_WGAN/improved_wgan_training-master-sort/samples/$1.png
