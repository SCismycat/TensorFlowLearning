#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.26 11:39

# Eager API可以简单理解为更加方便的构建图，不用在Session中执行。

from __future__ import absolute_import,division

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# set eager api
print("set eager mode..")
tfe.enable_eager_execution()

# 定义常量单元
print("define constant tensors..")
a = tf.constant(2)
b = tf.constant(3)

print("Running oprations without Session")
c = a+ b
print("a+b= %i" % c)
d = a * b
print("a*b=%i" % d)

print("mixing op with Tensors and Numpy arrays")
a = tf.constant([[2.,1.],
                 [1.,0.]],dtype=np.float32)
b = np.array([[3.,0.],
              [5.,1.]],dtype=np.float32)
c = a+b
print("a+b=%s" % c)
d = tf.matmul(a,b)
print("a*b=%s" % d)

print("Iterate through Tensor 'a'")
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])



