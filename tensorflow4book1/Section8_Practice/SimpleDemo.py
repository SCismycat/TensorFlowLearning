#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/18 17:52
# @Author  : Leslee

import tensorflow as tf
import  numpy as np

x_data = np.linspace(-1,1,300)[:,np.newaxis]

noise = np.random.normal(0,0.05,x_data.shape)

y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

def add_layer(inputs,in_size,out_size,activation_func= None):
    # 构建权重矩阵
    weight = tf.Variable(tf.random_normal([in_size,out_size]))
    # 构建偏置矩阵
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    y_ = tf.matmul(inputs,weight) + biases
    if activation_func is None:
        outputs = y_
    else:
        outputs = activation_func(y_)
    return outputs

# 构建隐藏层
h1 = add_layer(xs,1,20,activation_func=tf.nn.relu)
# 输出层
prediction = add_layer(h1,20,1,activation_func=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                     reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        print(sess.run([loss],feed_dict={xs:x_data,ys:y_data}))







