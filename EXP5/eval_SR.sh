#!/bin/bash
python cifar10_eval_master.py --checkpointdir /media/alan/SteinsGate/github_code/mnist/TensorFlow_WGAN/Cifar10/samples/$1/checkpoint
mv /media/alan/SteinsGate/github_code/mnist/TensorFlow_WGAN/Cifar10/samples/deNoisedMnist_samples.png /media/alan/SteinsGate/github_code/mnist/TensorFlow_WGAN/Cifar10/samples/$1.png
