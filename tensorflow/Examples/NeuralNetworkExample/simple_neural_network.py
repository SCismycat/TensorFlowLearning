#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.27 11:02

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network param
n_hidden_1 = 256
n_hidden_2 = 256
num_input = 784
num_classes = 10

X = tf.placeholder("float",[None,num_input])
Y = tf.placeholder("float",[None,num_classes])

# 存储weights和biases
weights = {
    'h1':tf.Variable(tf.random_normal([num_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,num_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# 创建模型
def neural_network(x):
    # 隐藏全连接层
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    # 隐藏层-- 第二层
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    # 输出层
    out_layers = tf.matmul(layer_2,weights['out']) + biases['out']
    return out_layers

# 构造模型
logits = neural_network(X)
prediction = tf.nn.softmax(logits) # 预测值

# 定义loss和optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)

# 评价模型
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# initial val
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for step in range(1,num_steps+1):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        # run optimizer op
        sess.run(train_op,feed_dict={X:batch_x,Y:batch_y})
        if step % display_step == 0 or step == 1:
            # 计算一个batch的loss和准确率
            loss, acc = sess.run([loss_op,accuracy],feed_dict={X:batch_x,Y:batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    print("Optimization Finished!")
    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: mnist.test.images,
                                        Y: mnist.test.labels}))
