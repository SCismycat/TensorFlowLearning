#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.29 14:39

import numpy as np
import tensorflower as tf
import time

# Import MNIST data
from tensorflower.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
num_gpus = 2
num_steps = 200
learning_rate = 0.001
batch_size = 1024
display_step = 10

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

def conv_net(x,n_classes,dropout,reuse,is_training):
    with tf.variable_scope('ConvNet',reuse=reuse):
        x = tf.reshape(x,shape=[-1,28,28,1])

        x = tf.layers.conv2d(x,64,5,activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x,2,2)

        x = tf.layers.conv2d(x,256,3,activation=tf.nn.relu)
        x = tf.layers.conv2d(x,512,3,activation=tf.nn.relu)

        x = tf.layers.max_pooling2d(x,2,2)

        x = tf.contrib.layers.flatten(x)
        # 全连接层
        x = tf.layers.dense(x,2048)
        x = tf.layers.dropout(x,rate=dropout,training=is_training)

        x = tf.layers.dense(x,1024)
        x = tf.layers.dropout(x,rate=dropout,training=is_training)

        out = tf.layers.dense(x,n_classes)
        out = tf.nn.softmax(out) if not is_training else out
    return out

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # 每个grad_and_vars参数都是(gpu0,梯度0)，....,(gpuN,梯度N)
        grads = []
        for g,_ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g,0)

            grads.append(expanded_g)

        # 对所有的梯度取平均
        grad = tf.concat(grads,0)
        grad = tf.reduce_mean(grad,0)

        # 由于参数变量在上面的list中是共享的，所以只返回第一个towers中的Variable
        v = grad_and_vars[0][1]
        grad_and_vars = (grad,v)
        average_grads.append(grad_and_vars)
    return average_grads

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']
def assign_to_device(device,ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op,tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

# 所有的Graph的操作默认放在CPU上
with tf.device('/cpu:0'):
    tower_grads = []
    reuse_vars = False
    # tf Graph input
    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])

    # 遍历所有GPU并构建自己的计算图
    for i in range(num_gpus):
        with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
            # Split data between GPUs
            _x = X[i * batch_size: (i + 1) * batch_size]
            _y = Y[i * batch_size: (i + 1) * batch_size]
            # 由于dropout在训练和预测阶段是不同的行为，所以需要创建两个独立的图来共享相同参数
            logits_train = conv_net(_x,num_classes,dropout,
                                    reuse=reuse_vars,is_training=True)

            logits_test = conv_net(_x,num_classes,dropout,
                                   reuse=True,is_training=False)

            # 定义loss和opts，带上logits_train 以使dropout生效
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train,labels=_y))
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads = optimizer.compute_gradients(loss_op)
            # 只用其中一个gpu计算acc
            if i ==0:
                # Evaluate model (with test logits, for dropout to be disabled)
                correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(_y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            reuse_vars = True
            tower_grads.append(grads)

    tower_grads = average_gradients(tower_grads)
    train_op = optimizer.apply_gradients(tower_grads)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for step in range(1,num_steps+1):
            batch_x, batch_y = mnist.train.next_batch(batch_size * num_gpus)
            ts = time.time()
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            te = time.time() - ts
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ": Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc) + ", %i Examples/sec" % int(len(batch_x) / te))
            step += 1
        print("Optimization Finished!")

        # Calculate accuracy for MNIST test images
        print("Testing Accuracy:",
              np.mean([sess.run(accuracy, feed_dict={X: mnist.test.images[i:i + batch_size],
                                                     Y: mnist.test.labels[i:i + batch_size]}) for i in
                       range(0, len(mnist.test.images), batch_size)]))



























