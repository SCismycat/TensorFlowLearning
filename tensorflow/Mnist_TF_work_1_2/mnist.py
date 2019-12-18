#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Leslee
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

NUM_CLASSES = 10

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
# 构建模型
def inference(images,hidden1_units,hidden2_units):
    # 第一层隐藏层
    with tf.compat.v1.name_scope('hidden1'):
        weights = tf.Variable(
            tf.random.truncated_normal([IMAGE_PIXELS,hidden1_units],stddev=1.0/math.sqrt(float(IMAGE_PIXELS)),name='weights')
        )
        biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images,weights) + biases)

    # 第二层隐藏层
    with tf.compat.v1.name_scope('hidden2'):
        weights = tf.Variable(
            tf.random.truncated_normal([hidden1_units,hidden2_units],stddev=1.0/math.sqrt(float(hidden1_units)),name='weights')
        )
        biases = tf.Variable(tf.zeros([hidden2_units]),name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1,weights) + biases)
    # 线性层，softmax
    with tf.compat.v1.name_scope('softmax_linear'):
        weights = tf.Variable(tf.random.truncated_normal([hidden2_units,NUM_CLASSES],stddev=1.0/math.sqrt(float(hidden2_units))),name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),name='biases')
        logits = tf.matmul(hidden2,weights) + biases

    return logits

def loss(logits,labels):
    labels = tf.cast(labels,dtype=tf.int64)
    return tf.compat.v1.losses.sparse_softmax_cross_entropy(
        labels=labels,logits=logits)

def training(loss,learning_rate):
    tf.compat.v1.summary.scalar('loss',loss)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0,name='global_step',trainable=True)
    train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op

def evaluation(logits,labels):
    correct = tf.nn.in_top_k(predictions=logits,targets=labels,k=1)
    return tf.reduce_sum(input_tensor=tf.cast(correct,tf.int32))




