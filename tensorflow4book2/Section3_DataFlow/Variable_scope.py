#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/24 17:34
# @Author  : Leslee

import tensorflow as tf

def my_img_filter(input_img):
    conv1_weights = tf.Variable(tf.random_normal([5,5,32,32]),
                                name='conv1_w')
    conv1_biases = tf.Variable(tf.zeros([32]),name='conv1_b')
    conv1 = tf.nn.conv2d(input_img,conv1_weights,strides=[1,1,1,1]
                         ,padding='SAME')
    relu1 = tf.nn.relu(conv1+conv1_biases)

    # 二层卷积
    conv2_weights = tf.Variable(tf.random_normal([5,5,32,32]),name='conv2_w')
    conv2_biases = tf.Variable(tf.zeros([32]),name='conv2_b')
    conv2 = tf.nn.conv2d(relu1,conv2_weights,strides=[1,1,1,1],padding='SAME')
    return tf.nn.relu(conv2+conv2_biases)
# 模型包含四个不同的参数，代码复杂度高。
# 模型复用存在问题。导致内存开销更大。

######################################
def conv_relu(input,kernel_shape,bias_shape):

    weights = tf.get_variable("weights",kernel_shape,
                              initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases",bias_shape,
                             initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input,weights,strides=[1,1,1,1],padding='SAME')
    return tf.nn.relu(conv+biases)

def my_img_filter_var_scope(input_imgs):
    with tf.variable_scope("conv1",reuse=True,initializer=tf.constant_initializer(0.3)):
        relu1 = conv_relu(input_imgs,[5,5,32,32],[32])

    with tf.variable_scope("conv2"):
        return conv_relu(relu1,[5,5,32,32],[32])