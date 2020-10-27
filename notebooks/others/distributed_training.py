"""Trains and Evaluates the MNIST network using a feed dictionary."""
import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


INPUT_DATA_DIR = '/tmp/tensorflow/mnist/input_data/'
#INPPUT_DATA_DIR = '/tmp/data/'
MAX_STEPS = 1000
BATCH_SIZE = 100
LEARNING_RATE = 0.3
HIDDEN_1 = 128
HIDDEN_2 = 32

# HACK: Ideally we would want to have a unique subpath for each instance of the job, but since we can't
# we are instead appending HOSTNAME to the logdir
LOG_DIR = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                       'tensorflow/mnist/logs/fully_connected_feed/', os.getenv('HOSTNAME', ''))

class TensorflowModel():
    def train(self, **kwargs):
        tf.logging.set_verbosity(tf.logging.ERROR)
        self.data_sets = input_data.read_data_sets(INPUT_DATA_DIR)
        self.images_placeholder = tf.placeholder(
            tf.float32, shape=(BATCH_SIZE, mnist.IMAGE_PIXELS))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE))

        logits = mnist.inference(self.images_placeholder,
                                 HIDDEN_1,
                                 HIDDEN_2)

        self.loss = mnist.loss(logits, self.labels_placeholder)
        self.train_op = mnist.training(self.loss, LEARNING_RATE)
        self.summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.summary_writer = tf.summary.FileWriter(LOG_DIR, self.sess.graph)
        self.sess.run(init)

        data_set = self.data_sets.train
        for step in xrange(MAX_STEPS):
            images_feed, labels_feed = data_set.next_batch(BATCH_SIZE, False)
            feed_dict = {
                self.images_placeholder: images_feed,
                self.labels_placeholder: labels_feed,
            }

            _, loss_value = self.sess.run([self.train_op, self.loss],
                                     feed_dict=feed_dict)
            if step % 100 == 0:
                print("At step {}, loss = {}".format(step, loss_value))
                summary_str = self.sess.run(self.summary, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, step)
                self.summary_writer.flush()

        
if __name__ == '__main__':
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow import fairing
        AWS_ACCOUNT_ID=fairing.cloud.aws.guess_account_id()
        AWS_REGION='us-west-2'
        DOCKER_REGISTRY = '{}.dkr.ecr.{}.amazonaws.com'.format(AWS_ACCOUNT_ID, AWS_REGION)

        fairing.config.set_preprocessor('python', input_files=[__file__])
        fairing.config.set_builder(name='append', registry=DOCKER_REGISTRY,
                           base_image='tensorflow/tensorflow:1.14.0-py3')
        fairing.config.set_deployer(
            name='tfjob', worker_count=1, ps_count=1)
        fairing.config.run()
    else:
        remote_train = TensorflowModel()
        remote_train.train()