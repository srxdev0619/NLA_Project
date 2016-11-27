from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
import scipy.misc
import scipy
import random
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

slim = tf.contrib.slim

losses = tf.contrib.losses

input_img = tf.placeholder(tf.float32, shape=[None, 784])

output = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(input_img, [-1,28,28,1])

net ={}
net['Conv1'] = slim.conv2d(x_image, 32, [5,5])
net['MaxPool1'] = slim.max_pool2d(net['Conv1'], [2,2])
net['Conv2'] = slim.conv2d(net['MaxPool1'], 64, [5,5])
net['MaxPool2'] = slim.max_pool2d(net['Conv2'], [2,2])
net['flatten'] = slim.flatten(net['MaxPool2'])
net['dropout'] = slim.dropout(net['flatten'], 0.8)
net['output'] = slim.fully_connected(net['dropout'], 10)

cross_entropy_loss = tf.reduce_mean(losses.softmax_cross_entropy(net['output'], output))

train_step = tf.train.AdamOptimizer().minimize(cross_entropy_loss)

correct_prediction = tf.equal(tf.argmax(net['output'], 1), tf.argmax(output, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

epochs = 200

initial_acc = 0.0

saver = tf.train.Saver()

with tf .Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(epochs):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={input_img: batch[0], output: batch[1]})
        val_accuracy = accuracy.eval(feed_dict={input_img: mnist.validation.images, output:mnist.validation.labels})
        if val_accuracy > initial_acc:
            initial_acc = val_accuracy
            saver.save(sess,"Mnist_NLA.ckpt")
        print("Current Validation accuracy is: ", val_accuracy*100,"%")
    test_accuracy = accuracy.eval(feed_dict={input_img:mnist.test.images, output:mnist.test.labels})
    print("The test accuracy is:", test_accuracy*100, "%")
    
        
