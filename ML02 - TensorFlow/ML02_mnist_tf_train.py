"""This is the training script.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import numpy as np
import tensorflow as tf
from ML02_mnist_tf_model import *
from ML02_mnist_tf_data import *

EVAL_CONST = 50

def eval_prediction(logits, labels):
    """Evaluates the prediction.

    Args:
        logits: A [batch_size, num_class] sized tensor.
        labels: A [batch_size] sized tensor.

    Returns:
        corrent_cnt: Counts the total number of correct prediction.
    """
    
    correct = tf.nn.in_top_k(logits, tf.to_int64(labels), 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def create_train_op(loss, learning_rate):
    """Creates a train operator to minimize the loss.

    Args:
        loss: Loss tensor.
        learning_rate: Float for gradient descent.

    Returns:
        train_op: The Op for training.
    """    
    # Creates the gradient descent optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # This variable is for tracking purpose.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Creates the minimization training op.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def train(config, train_data_set, valid_data_set=None):
    """Trains the feedfoward on MNIST for a number of steps"""
    with tf.Graph().as_default():
        # Generates input placeholders.
        images_placeholder, labels_placeholder = placeholder_inputs_feedfoward(
            config.batch_size, config.image_pixel_size
        )

        # Builds feedforward with two hidden layers.
        logits = feed_forward_net(images_placeholder, config)

        # Computes the loss using logits and labels.
        loss = compute_loss(logits, labels_placeholder)

        # Creates a training operator.
        train_op = create_train_op(loss, config.learning_rate)

        # Evals the current model by count the correct predictions.
        corrent_cnt_op = eval_prediction(logits, labels_placeholder)

        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep=1)

        with tf.Session() as sess:
            sess.run(init_op)

            for step in xrange(config.max_iters):
                print (step)
                # Fills in the inputs using fill_feed_dict().
                train_feed_dict = fill_feed_dict(
                    train_data_set, config.batch_size, images_placeholder,
                    labels_placeholder)

                # Runs the training operation.
                _, loss_val = sess.run([train_op, loss],
                                       feed_dict=train_feed_dict)

                if (step + 1) % EVAL_CONST == 0:
                    print('======Step {0}======'.format(step))
                    print('Train data evaluation:')
                    acc = evaluation(sess, images_placeholder,
                                     labels_placeholder,
                                     train_data_set, corrent_cnt_op)
                    print('train accuracy: {:.3f}'.format(acc))
                    if valid_data_set:
                        print('Validation data evlauation:')
                        acc = evaluation(sess, images_placeholder,
                                     labels_placeholder,
                                     valid_data_set, corrent_cnt_op)
                        print('valid accuracy: {:.3f}'.format(acc))
                    print('======')
                    # Saves the current model.
                    checkpoint_file = os.path.join(
                        config.model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)

            # Final evaluation when training is done.
            print('===Final data===')
            print('Train data evaluation:')
            acc = evaluation(sess, images_placeholder, labels_placeholder,
                             train_data_set, corrent_cnt_op)
            print('train accuracy: {:.3f}'.format(acc))
            if valid_data_set:
                print('Validation data evlauation:')
                acc = evaluation(sess, images_placeholder, labels_placeholder,
                                 valid_data_set, corrent_cnt_op)
                print('valid accuracy: {:.3f}'.format(acc))


def read_data():
    # Reads in training data.
    train_filename = './train_data.csv'
    train_images, train_labels = data_reader(train_filename)
    train_dataset = DataSet(train_images, train_labels)

    # Reads in validation data.
    valid_filename = './valid_data.csv'
    valid_images, valid_labels = data_reader(valid_filename)
    valid_dataset = DataSet(valid_images, valid_labels)
    return train_dataset, valid_dataset


def main():
    train_data_set, valid_data_set = read_data()
    config = Config()
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    train(config, train_data_set, valid_data_set=valid_data_set)


if __name__ == '__main__':
    main()
