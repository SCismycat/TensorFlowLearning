#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.27 16:39

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.001
training_step = 10000
batch_size = 128
display_step = 200

# network params
num_input = 28
timesteps = 28
num_hidden = 128
num_classes = 10

X = tf.placeholder("float",[None,timesteps,num_input])
Y = tf.placeholder("float",[None,num_classes])

# define weights
weights = {
    'out':tf.Variable(tf.random_normal([num_hidden,num_classes]))
}
biases = {
    'out':tf.Variable(tf.random_normal([num_classes]))
}

def RNN(x,weights,biases):
    # 首先需要把原数据的shape转换为rnn的输入，当前的输入shape是[batch_size,timesteps,n_inputs]
    # 需要的输入shape是 ‘timesteps’ tensor 的(batch_size,n_input)的list
    # 开始
    x = tf.unstack(x,timesteps,1)

    # Define 一个lstm cell
    lstm_cell = rnn.BasicLSTMCell(num_hidden,forget_bias=1.0)

    # 获取lstm cell的输出
    outputs,states = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
    # 线性激活，使用RNN内循环最后一个输出
    return tf.matmul(outputs[-1],weights['out']) + biases['out']

def BiRNN(x,weights,biases):
    x = tf.unstack(x,timesteps,1)
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden,forget_bias=1.0)

    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden,forget_bias=1.0)

    try:
        outputs, _,_ = rnn.stack_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)
    except Exception:
        outputs = rnn.stack_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)

    return tf.matmul(outputs[-1],weights['out']) + biases['out']


logits = RNN(X,weights,biases)
prediction = tf.nn.softmax(logits)

# 定义loss和optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)

# 评估模型
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,dtype=tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for step in range(1,training_step+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # 把数据reshape
        batch_x = batch_x.reshape((batch_size,timesteps,num_input))

        # 先run optimizer
        sess.run(train_op,feed_dict={X:batch_x,Y:batch_y})
        if step % display_step == 0 or step ==1:
            loss,acc = sess.run([loss_op,accuracy],feed_dict={X:batch_x,Y:batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    print("Optimizered Finished!")

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1,timesteps,num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
