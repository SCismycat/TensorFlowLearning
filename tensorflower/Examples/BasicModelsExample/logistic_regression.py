#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.26 14:29
import tensorflower as tf

from tensorflower.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

# paramters
learning_rate = 0.01
train_epochs = 25
batch_size = 100
display_step = 1

# 定义placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

# weights bias
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x,W)+b)

# 最小化交叉熵
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
# 梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化所有参数
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 运行初始化
    sess.run(init)

    for epoch in range(train_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 循环所有的batchs
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            #
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_xs,
                                                       y:batch_ys})
            # 计算平均损失
            avg_cost += c/total_batch
        if (epoch+1) % display_step ==0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Ooptimizer Finished!")

    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))

    # 计算正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))











