#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.29 14:26

import numpy as np
import tensorflower as tf
import datetime

log_device_placement = True

n = 10
# 创建随机大型矩阵
A = np.random.rand(10000,10000).astype('float32')
B = np.random.rand(10000,10000).astype('float32')
# 创建图存储结果
c1 = []
c2 = []

def matpow(M,n):
    if n<1:
        return M
    else:
        return tf.matmul(M,matpow(M,n-1))
"""单GPU情况"""
with tf.device('/gpu:0'):
    a = tf.placeholder(tf.float32,[10000,10000])
    b = tf.placeholder(tf.float32,[10000,10000])

    c1.append(matpow(a,n))
    c2.append(matpow(b,n))
with tf.device("/cpu:0"):
    sum = tf.add_n(c1)

t1_1 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    sess.run(sum,{a:A,b:B})
t2_1 = datetime.datetime.now()

"""多GPU情况"""
with tf.device('/gpu:0'):
    # Compute A^n and store result in c2
    a = tf.placeholder(tf.float32, [10000, 10000])
    c2.append(matpow(a, n))

with tf.device('/gpu:1'):
    b = tf.placeholder(tf.float32,[10000,10000])
    c2.append(matpow(b,n))

with tf.device('/cpu:0'):
    sum = tf.add_n(c2)

t1_2 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    sess.run(sum,{a:A,b:B})
t2_2 = datetime.datetime.now()

print("Single GPU computation time: " + str(t2_1-t1_1))
print("Multi GPU computation time: " + str(t2_2-t1_2))
