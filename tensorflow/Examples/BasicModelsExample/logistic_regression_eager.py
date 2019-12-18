#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.26 14:48

from __future__ import absolute_import,division
import tensorflow as tf

tf.enable_eager_execution()
tfe = tf.contrib.eager

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# Parameters
learning_rate = 0.1
batch_size = 128
num_steps = 1000
display_step = 100

dataset = tf.data.Dataset.from_tensor_slices(
    (mnist.train.images,mnist.train.labels))
dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
dataser_iter = tfe.Iterator(dataset)

W = tfe.Variable(tf.zeros([784,10]),name='weights')
b = tfe.Variable(tf.zeros([10]),name='bias')
# 定义一个回归函数
def logistic_regerssion(inputs):
    return tf.matmul(inputs,W) + b
# 定义损失函数，回归函数作为参数传入
def loss_fn(interface_fn,inputs,labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=interface_fn(inputs),labels=labels))
# calculate accuarcy
def accuarcy_fn(interface_fn,inputs,labels):
    prediction = tf.nn.softmax(interface_fn(inputs))
    correct_pred = tf.equal(tf.argmax(prediction,1),labels) # 计算预测值和标签是否相等
    return tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# SGD optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# 计算梯度
grad = tfe.implicit_gradients(loss_fn)

# training
average_loss = 0.
average_acc = 0.
for step in range(num_steps):
    d = dataser_iter.next()
    # Images
    x_batch = d[0]
    # labels
    y_batch = tf.cast(d[1],dtype=tf.int64)

    # 计算一个batch 的loss
    batch_loss = loss_fn(logistic_regerssion,x_batch,y_batch)
    average_loss += batch_loss
    # 计算一个batch 的正确率
    batch_accuracy = accuarcy_fn(logistic_regerssion,x_batch,y_batch)

    if step == 0:
        print("initial loss={:.9f}".format(average_loss))

    # 根据梯度变化更新变量
    optimizer.apply_gradients(grad(logistic_regerssion,x_batch,y_batch))

    #
    if (step + 1) % display_step == 0 or step == 0:
        if step > 0:
            average_loss /= display_step
            average_acc /= display_step
        print("Step:", '%04d' % (step + 1), " loss=",
              "{:.9f}".format(average_loss), " accuracy=",
              "{:.4f}".format(average_acc))
        average_loss = 0.
        average_acc = 0.


test_X = mnist.test.images
test_Y = mnist.test.labels

test_acc = accuarcy_fn(logistic_regerssion,test_X,test_Y)
print("Testing accuracy:{:.4f}".format(test_acc))






































