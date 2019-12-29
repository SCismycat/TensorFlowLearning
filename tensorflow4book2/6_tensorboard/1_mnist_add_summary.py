#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/25 17:00
# @Author  : Leslee

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/MNIST_data",one_hot=True)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784],name='x_input')
    y = tf.placeholder(tf.float32,[None,10],name='y_input')

with tf.name_scope('softmax_layer'):
  with tf.name_scope('weights'):
      weights = tf.Variable(tf.zeros([784, 10]))
  with tf.name_scope('biases'):
      biases = tf.Variable(tf.zeros([10]))
  with tf.name_scope('Wx_plus_b'):
      y = tf.matmul(x, weights) + biases

with tf.name_scope('cross_entropy'):
  diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(diff)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(0.001).minimize(
      cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
writer = tf.summary.FileWriter('../model/summary/mnist', sess.graph)
tf.global_variables_initializer().run()

writer.close()