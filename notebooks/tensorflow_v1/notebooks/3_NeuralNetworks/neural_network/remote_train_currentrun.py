from __future__ import division, print_function, absolute_import
import argparse

parser = argparse.ArgumentParser(description='Example with non-optional arguments')

parser.add_argument('method', action="store")
parser.add_argument('dataDir', action="store")
parser.add_argument('baseImage', action="store")
parser.add_argument('entryPoint', action="store")
args = parser.parse_args()
method = args.method
dataDir = args.dataDir
baseImage = args.baseImage
entryPoint = args.entryPoint
print (parser.parse_args())
#from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# Parameters
learning_rate = 0.1
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer
# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    
    # Build the neural network
    logits = neural_net(features)
    
    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)
    
    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes) 
        
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    
    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    
    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=pred_classes,
      loss=loss_op,
      train_op=train_op,
      eval_metric_ops={'accuracy': acc_op})

    return estim_specs
# Build the Estimator
model = tf.estimator.Estimator(model_fn)
# Train the Model
model.train(input_fn, steps=num_steps)
import random, string
import os
import subprocess
import importlib
from kubeflow import fairing
from kubeflow.fairing import TrainJob
from kubeflow.fairing.backends import KubeflowAWSBackend

AWS_REGION = 'us-west-2'
FAIRING_BACKEND = 'KubeflowAWSBackend'
AWS_ACCOUNT_ID = fairing.cloud.aws.guess_account_id()
BASE_DOCKER_IMAGE = baseImage
DOCKER_REGISTRY = '{}.dkr.ecr.{}.amazonaws.com'.format(AWS_ACCOUNT_ID, AWS_REGION)
#S3_BUCKET = f'{HASH}-kubeflow-pipeline-data'
S3_BUCKET = 'pjz16s-eks-ml-data'
ENTRY_POINT = entryPoint


#NOTEBOOK_BASE_DIR = fairing.notebook.notebook_util.get_notebook_name()

DATASET = dataDir
REQUIREMENTS = 'requirements.txt'


if FAIRING_BACKEND == 'KubeflowAWSBackend':
    from kubeflow.fairing.builders.cluster.s3_context import S3ContextSource
    BuildContext = S3ContextSource(
        aws_account=AWS_ACCOUNT_ID, region=AWS_REGION,
        bucket_name=S3_BUCKET
    )

BackendClass = getattr(importlib.import_module('kubeflow.fairing.backends'), FAIRING_BACKEND)

print("About to train job setup...")
from kubeflow.fairing import TrainJob
train_job = TrainJob(ENTRY_POINT, input_files=[DATASET,REQUIREMENTS],
                     base_docker_image=BASE_DOCKER_IMAGE,
                     docker_registry=DOCKER_REGISTRY,
                     backend=BackendClass(build_context_source=BuildContext))
print("about to submit job")
train_job.submit()