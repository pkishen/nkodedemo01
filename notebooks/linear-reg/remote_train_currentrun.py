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
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random
# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50
# Training Data
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]
# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")
# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)
# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
# Start training
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print ("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
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