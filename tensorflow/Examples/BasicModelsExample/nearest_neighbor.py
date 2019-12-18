#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.26 16:03

# 最近邻算法
import numpy as np
import tensorflow as tf

#
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.test.next_batch(200)

xtr = tf.placeholder("float",[None,784])
xte = tf.placeholder("float",[784])

# 使用l1距离计算最近邻
# 计算L1距离，损失函数
distance = tf.reduce_sum(tf.abs(tf.add(xtr,tf.negative(xte))),reduction_indices=1)
# 预测：找到最小距离索引,
pred = tf.arg_min(distance,0)

accuracy = 0.

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    for i in range(len(Xte)):
        # 获取最近邻数据
        nn_idx = sess.run(pred,feed_dict={xtr:Xtr,xte:Xte[i,:]})

        # 获取最近邻类别标签，并且和真正的标签进行比较
        print("Test",i,"Prediction:",np.argmax(Ytr[nn_idx]),"True class:",np.argmax(Yte[i]))

        if np.argmax(Ytr[nn_idx]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print("done!")
    print("accuracy",accuracy)