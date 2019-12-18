#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.26 11:22

import tensorflow as tf
# 常量操作方式
a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print("Add constant: %i" % sess.run(a+b))
    print("multy constant: %i" % sess.run(a*b))

# 变量表示方式
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a,b)
mul = tf.multiply(a,b)

with tf.Session() as sess:
    print("add with variables: %i" % sess.run(add,feed_dict={a:2,b:3}))
    print("multi with variables: %i" % sess.run(mul,feed_dict={a:3,b:4}))

# 矩阵乘法的计算方法
matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    res = sess.run(product)
    print(res)






