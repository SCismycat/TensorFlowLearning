#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/19 14:06
# @Author  : Leslee


import tensorflow as tf

a = tf.constant([1,1])
b = tf.constant([2,2])
c = tf.add(a,b)

# sparseTensor
sp = tf.SparseTensor(indices=[[0,2],[1,3]],values=[1,2],dense_shape=[3,4])
# 表示以dense_shape为形状的，稀疏向量。indices表示位置，第0行的第二个元素和第一行的
# 第三个元素，values分别是1,2
#[[0,0,1,0]
# [0,0,0,2]
# [0,0,0,0]]

reduce_x = [tf.sparse_reduce_sum(sp),
            tf.sparse_reduce_sum(sp,axis=1),
            tf.sparse_reduce_sum(sp,axis=[0,1])]

with tf.Session() as sess:
    print("a[0]=%s, a[1]=%s" % (a[0].eval(), a[1].eval())) # 运行结果
    print("c.name=%s" % c.name) # 在图中的层级名称
    print("c.shape=%s" % c.shape) # shape
    print("a.consumers=%s" % a.consumers()) # 后置操作
    print("c.op: %s" % c.op)
    # sparseTensor
    print(sess.run(reduce_x))