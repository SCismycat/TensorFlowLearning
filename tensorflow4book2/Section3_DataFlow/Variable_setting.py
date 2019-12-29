#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/24 15:21
# @Author  : Leslee
import tensorflow as tf
# 很多种初始化方法
W = tf.Variable(initial_value=tf.random_normal(shape=(1,4),mean=100,stddev=0.35),name="W")

w_replica = tf.Variable(W.initialized_value(),name='w_re')

w_twice = tf.Variable(W.initialized_value()*2.0,name='w_twice')

b = tf.Variable(tf.zeros([4]),name='biases')
# 不能直接使用变量实例作为初始化，会在session执行过程中产生循环依赖。
tf.global_variables_initializer
#部分变量初始化tf.variables_initializer()
with tf.Session() as sess:
    sess.run(tf.variables_initializer([W]))