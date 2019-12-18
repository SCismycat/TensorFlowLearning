#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.29 9:56

from tensorflower.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

import tensorflower as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

# 模型参数
num_hidden_1 = 256
num_hidden_2 = 128
num_inputs = 784

X = tf.placeholder("float",[None,num_inputs])

weights = {
    'encoder_h1':tf.Variable(tf.random_normal([num_inputs,num_hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2])),
    'decoder_h1':tf.Variable(tf.random_normal([num_hidden_2,num_hidden_1])),
    'decoder_h2':tf.Variable(tf.random_normal([num_hidden_1,num_inputs])),
}
biases = {
    'encoder_b1':tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2':tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1':tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2':tf.Variable(tf.random_normal([num_inputs])),
}

# encoder
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['decoder_h2']),biases['decoder_b2']))

    return layer_2

# 构建图的操作
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# prediction
y_pred = decoder_op
y_true = X

loss = tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    for i in range(1,num_steps+1):
        batch_x, _ = mnist.train.next_batch(batch_size)
        # 运行optimizer op(反向传播) 和 loss
        _, l = sess.run([optimizer,loss],feed_dict={X:batch_x})

        if i % display_step == 0 or i ==1:
            print('Step %i: Minibatch Loss: %f' % (i, l))
    # Testing
    n = 4
    canvas_orig = np.empty((28*n,28*n))
    canvas_recon = np.empty((28*n,28*n))
    for i in range(n):
        batch_x, _ = mnist.test.next_batch(n)

        g = sess.run(decoder_op,feed_dict={X:batch_x})
        # 打印原有图像
        for j in range(n):
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # 打印预测图像
        for j in range(n):
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

print("Origin images")
plt.figure(figsize=(n,n))
plt.imshow(canvas_orig,origin="upper",cmap="gray")
plt.show()

print("Reconstructed Images")
plt.figure(figsize=(n,n))
plt.imshow(canvas_recon,origin="upper",cmap="gray")
plt.show()























