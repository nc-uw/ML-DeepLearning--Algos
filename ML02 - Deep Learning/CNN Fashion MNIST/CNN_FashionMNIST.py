# Import libraries
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split

'''
# Import Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
x_train = fashion_mnist[0][0]   
y_train = fashion_mnist[0][1]
y_test = fashion_mnist[1][0]
y_test = fashion_mnist[1][1]
# Download from link    
data = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
'''
    
data = input_data.read_data_sets('data/fashion', one_hot=True)

print("Training set (images) shape: {shape}".format(shape=data.train.images.shape))
print("Training set (labels) shape: {shape}".format(shape=data.train.labels.shape))
print("Test set (images) shape: {shape}".format(shape=data.test.images.shape))
print("Test set (labels) shape: {shape}".format(shape=data.test.labels.shape))

label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot'
}

plt.figure(figsize=[5,5])
plt.subplot(121)
curr_img = np.reshape(data.train.images[0], (28,28))
curr_lbl = np.argmax(data.train.labels[0,:])
plt.imshow(curr_img)
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

plt.subplot(122)
curr_img = np.reshape(data.test.images[0], (28,28))
curr_lbl = np.argmax(data.test.labels[0,:])
plt.imshow(curr_img)
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

train_X = data.train.images.reshape(-1, 28, 28, 1)
test_X = data.test.images.reshape(-1,28,28,1)

train_X = data.train.images.reshape(-1, 28, 28, 1)
test_X = data.test.images.reshape(-1,28,28,1)

train_y = data.train.labels
test_y = data.test.labels

training_iters = 3
learning_rate = 0.001 
batch_size = 256

n_input = 28
n_classes = 10

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv_net(x, weights, biases, drp, bnorm_flg):  

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    if bnorm_flg == 1:
        conv1_mean, conv1_var = tf.nn.moments(conv1, axes=[0, 1, 2], name='moments')
        conv1 = tf.nn.batch_normalization(conv1, conv1_mean, conv1_var, None, None, 1e-3)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    if bnorm_flg == 1:
        conv2_mean, conv2_var = tf.nn.moments(conv2, axes=[0, 1, 2], name='moments')
        conv2 = tf.nn.batch_normalization(conv2, conv2_mean, conv2_var, None, None, 1e-3)
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)
    if bnorm_flg == 1:
        conv3_mean, conv3_var = tf.nn.moments(conv3, axes=[0, 1, 2],  name='moments')
        conv3 = tf.nn.batch_normalization(conv3, conv3_mean, conv3_var, None, None, 1e-3)
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    if bnorm_flg == 1:
        fc1_mean, fc1_var = tf.nn.moments(fc1, axes=[0], name='moments')
        fc1 = tf.nn.batch_normalization(fc1, conv1_mean, conv1_var, None, None, 1e-3)
    if drp > 0.:
        fc1 = tf.nn.dropout(fc1, keep_prob=drp)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def trainz(drp, bnorm_flg):
    print ('\nTraining for droput = ', drp)
    pred = conv_net(x, weights, biases, drp, bnorm_flg)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init) 
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        summary_writer = tf.summary.FileWriter('./Output', sess.graph)
        for i in range(training_iters):
            for batch in range(len(train_X)//batch_size):
                batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
                batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    
                opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(i) + ", Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))
            print("Optimization Finished!")
    
            test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
            train_loss.append(loss)
            test_loss.append(valid_loss)
            train_accuracy.append(acc)
            test_accuracy.append(test_acc)
            print("Testing Accuracy:","{:.5f}".format(test_acc))
        summary_writer.close()
    
    
    plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
    plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
    plt.title('Training and Test loss')
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()
    
    plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
    plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()
    

print ('\n\nTraining without BatchNorm')
for drp in list(np.arange(0.0,0.9,0.1)):
    
    tf.reset_default_graph()
    x = tf.placeholder("float", [None, 28,28,1])
    y = tf.placeholder("float", [None, n_classes])
    weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(3,3,64,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,64,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W3', shape=(4*4*64,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W6', shape=(64,n_classes), initializer=tf.contrib.layers.xavier_initializer())}

    biases = {
    'bc1': tf.get_variable('B0', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer())}

    trainz(drp,bnorm_flg=0)

print ('\n\nTraining with BatchNorm')
tf.reset_default_graph()
x = tf.placeholder("float", [None, 28,28,1])
y = tf.placeholder("float", [None, n_classes])
weights = {
'wc1': tf.get_variable('W0', shape=(3,3,1,64), initializer=tf.contrib.layers.xavier_initializer()), 
'wc2': tf.get_variable('W1', shape=(3,3,64,64), initializer=tf.contrib.layers.xavier_initializer()), 
'wc3': tf.get_variable('W2', shape=(3,3,64,64), initializer=tf.contrib.layers.xavier_initializer()), 
'wd1': tf.get_variable('W3', shape=(4*4*64,64), initializer=tf.contrib.layers.xavier_initializer()), 
'out': tf.get_variable('W6', shape=(64,n_classes), initializer=tf.contrib.layers.xavier_initializer())}

biases = {
'bc1': tf.get_variable('B0', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
'bc3': tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
'bd1': tf.get_variable('B3', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer())}
trainz(drp=0.,bnorm_flg=1)
