#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.27 14:54

import tensorflower as tf

# Import MNIST data
from tensorflower.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 200
batch_size = 128
display_step = 10

# network params
num_input = 784
num_classes = 10
dropout = 0.75

# tf Graph input
X = tf.placeholder(tf.float32,[None,num_input])
Y = tf.placeholder(tf.float32,[None,num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout settings

# 创建卷积神经网络的每个层
def conv2d(x,W,b,strides=1):

    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

# 创建模型
def conv_network(x,weights,biases,dropout):
    # mnist是1-D的784维的向量，reshape维度为[Height*Width*depth]
    # Tensor变成4-D的向量，即[batch_size,height,width,depth]
    x = tf.reshape(x,shape=[-1,28,28,1])

    # j卷积层
    conv1 = conv2d(x,weights['wc1'],biases['bc1'])
    # max pooling
    conv1 = maxpool2d(conv1,k=2)

    # 卷积层
    conv2 = conv2d(conv1,weights['wc2'],biases['bc2'])
    conv2 = maxpool2d(conv2,k=2)

    # 全连接层
    # 把conv2的维度reshape成全连接层的输入,拉平
    fc1 = tf.reshape(conv2,[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Dropout
    fc1 = tf.nn.dropout(fc1,dropout)
    out = tf.add(tf.matmul(fc1,weights['out']),biases['out'])
    return out

# 保存w&b
weights = {
    # 5*5的conv,1 input,32 output
    'wc1':tf.Variable(tf.random_normal([5,5,1,32])),
    # 5*5的conv ,32 input,64 output
    'wc2':tf.Variable(tf.random_normal([5,5,32,64])),
    # 全连接层，7*7*64 inputs,1024 outputs
    'wd1':tf.Variable(tf.random_normal([7*7*64,1024])),
    'out':tf.Variable(tf.random_normal([1024,num_classes]))
}

biases = {
    'bc1':tf.Variable(tf.random_normal([32])),
    'bc2':tf.Variable(tf.random_normal([64])),
    'bd1':tf.Variable(tf.random_normal([1024])),
    'out':tf.Variable(tf.random_normal([num_classes]))
}

# 构建模型
logits = conv_network(X,weights,biases,keep_prob)
prediction = tf.nn.softmax(logits)

# 定义loss和optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)


# 构建评价模型
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# init
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for step in range(1,num_steps+1):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        # 跑优化器，也就是反向传播
        sess.run(train_op,feed_dict={X:batch_x,Y:batch_y,keep_prob:0.8})
        if step % display_step == 0 or step == 1:
            loss,acc = sess.run([loss_op,accuracy],feed_dict={X:batch_x,Y:batch_y,keep_prob:1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    print("Optimizer Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                        Y: mnist.test.labels[:256],
                                        keep_prob: 1.0}))
