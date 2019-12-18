#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.26 14:05

from __future__ import absolute_import,division

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# set Eager API
tf.enable_eager_execution()
tfe = tf.contrib.eager

train_X = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
           7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]
train_Y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
           2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]
n_samples = len(train_X)

# Parameters
learning_rate = 0.01
display_step = 100
num_steps = 1000

# weight bias
W = tfe.Variable(np.random.randn())
b = tfe.Variable(np.random.randn())

# 线性回归函数
def linear_regression(inputs):
    return inputs * W + b

# 均方损失误差
def mean_square_fn(model_fn, inputs, labels):
    return tf.reduce_sum(tf.pow(model_fn(inputs)-labels,2)) / (2*n_samples)

# SGD optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# 计算梯度
grad = tfe.implicit_gradients(mean_square_fn)

# 在进行优化函数之前，先初始化损失函数
print("initial cost={:.9f}".format(
    mean_square_fn(linear_regression,train_X,train_Y)),
    "W=",W.numpy(),"b=",b.numpy())

# Training

for step in range(num_steps):
    optimizer.apply_gradients(grad(linear_regression,train_X,train_Y))

    if (step+1) % display_step == 0 or step==0:
        print("Epoch:", '%04d' % (step + 1), "cost=",
              "{:.9f}".format(mean_square_fn(linear_regression, train_X, train_Y)),
              "W=", W.numpy(), "b=", b.numpy())

plt.plot(train_X,train_Y,'ro',label='Original data')
plt.plot(train_X,np.array(W*train_X+b),label='Fitting line')
plt.legend()
plt.show()














