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
import os
import sys
import tensorflow as tf
import numpy as np

def nkTrain():
    # Genrating random linear data 
    # There will be 50 data points ranging from 0 to 50 
    x = np.linspace(0, 50, 50) 
    y = np.linspace(0, 50, 50) 

    # Adding noise to the random linear data 
    x += np.random.uniform(-4, 4, 50) 
    y += np.random.uniform(-4, 4, 50) 

    n = len(x) # Number of data points 

    X = tf.placeholder("float") 
    Y = tf.placeholder("float")
    W = tf.Variable(np.random.randn(), name = "W") 
    b = tf.Variable(np.random.randn(), name = "b") 
    learning_rate = 0.01
    training_epochs = 1000
    
    # Hypothesis 
    y_pred = tf.add(tf.multiply(X, W), b) 

    # Mean Squared Error Cost Function 
    cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n)

    # Gradient Descent Optimizer 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

    # Global Variables Initializer 
    init = tf.global_variables_initializer() 


    sess = tf.Session()
    sess.run(init) 
      
    # Iterating through all the epochs 
    for epoch in range(training_epochs): 
          
        # Feeding each data point into the optimizer using Feed Dictionary 
        for (_x, _y) in zip(x, y): 
            sess.run(optimizer, feed_dict = {X : _x, Y : _y}) 
          
        # Displaying the result after every 50 epochs 
        if (epoch + 1) % 50 == 0: 
            # Calculating the cost a every epoch 
            c = sess.run(cost, feed_dict = {X : x, Y : y}) 
            print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b)) 
      
    # Storing necessary values to be used outside the Session 
    training_cost = sess.run(cost, feed_dict ={X: x, Y: y}) 
    weight = sess.run(W) 
    bias = sess.run(b) 

    print('Weight: ', weight, 'Bias: ', bias)
import random, string
import os
import logging
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
#train_job = TrainJob(str(ENTRY_POINT), input_files=[DATASET,REQUIREMENTS],
#                     base_docker_image=BASE_DOCKER_IMAGE,
#                     docker_registry=DOCKER_REGISTRY,
#                     backend=BackendClass(build_context_source=BuildContext))

print('ENTRY POINT for building image is :'+ ENTRY_POINT)


if ENTRY_POINT == "nkTrain" :
    print('Now in FUNCTION  or CLASS entrypoint. Entry point is set as '+ENTRY_POINT)
    train_job = TrainJob(nkTrain, input_files=[DATASET,REQUIREMENTS],
                     base_docker_image=BASE_DOCKER_IMAGE,
                     docker_registry=DOCKER_REGISTRY,
                     backend=BackendClass(build_context_source=BuildContext))
else :
    print('Now in NON function or class entrypoint. Entry point is set as '+ENTRY_POINT)
    train_job = TrainJob(str(ENTRY_POINT), input_files=[DATASET,REQUIREMENTS],
                     base_docker_image=BASE_DOCKER_IMAGE,
                     docker_registry=DOCKER_REGISTRY,
                     backend=BackendClass(build_context_source=BuildContext))




print("about to submit job")

outJob = train_job.submit()

print ("Job to create image  is :  " + str(outJob))
print("Successfully created Docker Image in the configured registry "+ DOCKER_REGISTRY)
