#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.26 13:32

# 使用Tensorflow实现逻辑回归学习

import tensorflower as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# 训练数据
train_x =  numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_samples = train_x.shape[0]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(rng.randn(),name="weights")
b = tf.Variable(rng.randn(),name="bias")

# 创建一个线性模型
pred = tf.add(tf.multiply(X,W),b)

# 设置损失函数（均方损失函数）
cost = tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)

# 梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# init
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:
    # 运行初始化
    sess.run(init)

    # 填充所有训练数据
    for epoch in range(training_epochs):
        for (x,y) in zip(train_x,train_y):
            sess.run(optimizer,feed_dict={X:x,Y:y})

        # 打印日志
        if (epoch+1) % display_step == 0:
            c = sess.run(cost,feed_dict={X:train_x,Y:train_y})
            print("Epoch:",'%04d' % (epoch+1),"cost=","{:.9f}".format(c),"W=",sess.run(W),"b=",sess.run(b))

    print("optimization Finished！")
    training_cost = sess.run(cost,feed_dict={X:train_x,Y:train_y})
    print("Training cost=",training_cost,"W=",sess.run(W),"b=",sess.run(b),'\n')

    plt.plot(train_x,train_y,'ro',label="origin data")
    plt.plot(train_x,sess.run(W)*train_x+sess.run(b),label="fitting data")
    plt.legend()
    plt.show()

    # 测试数据
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing mean square loss")
    testing_cost = sess.run(tf.reduce_sum(tf.pow(pred-Y,2))/(2*test_X.shape[0]),
                            feed_dict={X:test_X,Y:test_Y})
    print("Testing cost=",testing_cost)
    print("absoluate mean square loss difference:",abs(training_cost-testing_cost))

    plt.plot(test_X,test_Y,'bo',label='Test data')
    plt.plot(train_x,sess.run(W)*train_x + sess.run(b),label='Fitted line')
    plt.legend()
    plt.show()












































