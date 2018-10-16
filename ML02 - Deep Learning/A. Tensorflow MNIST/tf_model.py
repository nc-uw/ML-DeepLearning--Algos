"""This contains model functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import tensorflow as tf

class Config(object):
    """This is a wrapper for all configurable parameters for model.

    Attributes:
        batch_size: Integer for the batch size.
        learning_rate: Float for the learning rate.
        image_pixel_size: Integer for the flatten image size.
        hidden1_size: Integer for the 1st hidden layer size.
        hidden2_size: Integer for the 2nd hidden layer size.
        num_class: Integer for the number of label classes.
        max_iters: Integer for the number of training iterations.
        model_dir: String for the output model dir.
    """

    def __init__(self):
        self.batch_size = 100
        self.learning_rate = 1e-3
        # Each image is 28x28.
        self.image_pixel_size = 784
        self.hidden1_size = 128
        self.hidden2_size = 128
        self.num_class = 10
        self.max_iters = 400
        self.model_dir = './model'

def placeholder_inputs_feedfoward(batch_size, feat_dim):
    """Creats the input placeholders for the feedfoward neural network.

    Args:
        batch_size: Integer for the batch size.
        feat_dim: Integer for the feature dimension.

    Returns:
        image_placeholder: Image placeholder.
        label_placeholder: Label placeholder.
    """
    image_placeholder = tf.placeholder(tf.float32, shape=(None,feat_dim))
    label_placeholder = tf.placeholder(tf.bool, shape=None)
    
    return image_placeholder, label_placeholder


def fill_feed_dict(data_set, batch_size, image_ph, label_ph):
    """Given the data for current step, fills both placeholders.

    Args:
        data_set: The DataSet object.
        batch_size: Integer for the batch size.
        image_ph: The image placeholder, from placeholder_inputs_feedfoward().
        label_ph: The label placehodler, from placeholder_inputs_feedfoward().

    Returns:
        feed_dict: The feed dictionary maps from placeholders to values.
    """

    image_feed, label_feed = data_set.next_batch(batch_size)
    
    feed_dict = {
        image_ph: image_feed,
        label_ph: label_feed,
    }
    
    return feed_dict


def feed_forward_net(images, config):
    """Creates a feedforward neuralnetwork.

    Args:
        images: Image placeholder.
        config: The Config object contains model parameters.

    Returns:
        logits: Output tensor with logits.
    """
    #Creates the 1st feed fully-connected layer with ReLU activation.
    with tf.variable_scope('hidden_layer_1'):
        # Creates two variables:
        # 1) hidden1_weights with size [image_pixel_size, hidden1_size].
        # 2) hidden1_biases with size [hidden1_size].
        #check for images variable
        weights = tf.Variable(tf.truncated_normal([config.image_pixel_size, config.hidden1_size], 
                                                   stddev=1.0 / math.sqrt(float(config.image_pixel_size))),name='weights')
        biases = tf.Variable(tf.zeros([config.hidden1_size]),name='biases')
        # Performs feedforward on images using the two variables defined above.
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
        #dropout prob = 1
        hidden1_dropout = tf.nn.dropout(hidden1, 1)
        #hidden1_dropout=hidden1
        reg_1=tf.nn.l2_loss(weights)
        #reg_1=0
    #Creates the 2nd feed fully-connected layer with ReLU activation.
    with tf.variable_scope('hidden_layer_2'):
        # Creates two variables:
        # 1) hidden2_weights with size [hidden1_size, hidden2_size].
        # 2) hidden2_biases with size [hidden2_size].
        weights = tf.Variable(tf.truncated_normal([config.hidden1_size, config.hidden2_size],
                                                  stddev=1.0 / math.sqrt(float(config.hidden1_size))),name='weights')
        biases = tf.Variable(tf.zeros([config.hidden2_size]),name='biases')
        # Performs feedforward on hidden1 using the two variables defined above.
        hidden2 = tf.nn.relu(tf.matmul(hidden1_dropout, weights) + biases)
        hidden2_dropout = tf.nn.dropout(hidden2, 1)
        #hidden2_dropout=hidden2
        reg_2=tf.nn.l2_loss(weights)
        #reg_2=0
    #Creates the pen-ultimate linear layer.
    with tf.variable_scope('logits_layer'):
        # Creates two variables:
        # 1) logits_weights with size [config.hidden2_size, config.num_class].
        # 2) logits_biases with size [config.num_class].
        weights = tf.Variable(tf.truncated_normal([config.hidden2_size, config.num_class], 
                                                  stddev=1.0 / math.sqrt(float(config.hidden2_size))),name='weights')
        biases = tf.Variable(tf.zeros([config.num_class]),name='biases')
        # Performs linear projection on hidden2 using the two variables above.
        logits = tf.matmul(hidden2_dropout, weights) + biases
        reg_3=tf.nn.l2_loss(weights)
        #reg_3=0
    return logits, reg_1+reg_2+reg_3


def compute_loss(logits, labels, reg):
    """Computes the cross entropy loss between logits and labels.

    Args:
        logits: A [batch_size, num_class] sized float tensor.
        labels: A [batch_size] sized integer tensor.
	 reg: A [batch_size] sized float tensor.

    Returns:
        loss: Loss tensor.
    """

    #Computes the cross-entropy loss.
    #labels = tf.to_int64(labels)
    #0.01= regularisation param
    loss = 0.01 * reg + tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.to_int64(labels), logits=logits)
    return loss

def evaluation(sess, image_ph, label_ph, data_set, eval_op):
    """Runs one full evaluation and computes accuracy.

    Args:
        sess: The session object.
        image_ph: The image placeholder.
        label_ph: The label placeholder.
        data_set: The DataSet object.
        eval_op: The evaluation accuracy op.

    Returns:
        accuracy: Float scalar for the prediction accuracy.
    """
    ##config=Config()

    #Computes the accuracy.
    feed_dict = fill_feed_dict(data_set, data_set.num_samples, image_ph, label_ph)
    count=sess.run(eval_op, feed_dict=feed_dict)
    accuracy=count/data_set.num_samples
    print('Num samples: %d  Num correct: %d  Accuracy: %0.02f' %
         (data_set.num_samples, count, accuracy))
    return accuracy
