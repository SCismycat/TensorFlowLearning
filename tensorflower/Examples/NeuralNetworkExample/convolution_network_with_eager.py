#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.27 15:58

from tensorflower.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=False)

import tensorflower as tf

# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 128

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit

# 创建神经网络
def conv_network(x_dict,n_classes,dropout,reuse,is_training):

    with tf.variable_scope('ConvNetwork',reuse=reuse):
        x = x_dict['images']

        x = tf.reshape(x,shape=[-1,28,28,1])

        conv1 = tf.layers.conv2d(x,32,5,activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1,2,2,padding='SAME')

        conv2 = tf.layers.conv2d(conv1,64,3,activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2,2,2)
        # 全连接层，需要把上一个输入拉平
        fc1 = tf.contrib.layers.flatten(conv2)

        fc1 = tf.layers.dense(fc1,1024)

        fc1 = tf.layers.dropout(fc1,rate=dropout,training=is_training)
        out = tf.layers.dense(fc1,n_classes)
    return out

# 用TF Estimator 模板定义模型函数
def model_fn(features,labels,mode):
    logits_train = conv_network(features,num_classes,dropout,reuse=False,is_training=True)
    logits_test = conv_network(features,num_classes,dropout,reuse=True,is_training=False)

    pred_classes = tf.argmax(logits_test,axis=1)
    pred_probs = tf.nn.softmax(logits_test)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train,labels=tf.cast(labels,dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # 验证模型的acc
    acc_op = tf.metrics.accuracy(labels=labels,predictions=pred_classes)

    estim_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy':acc_op})

    return estim_spec

model = tf.estimator.Estimator(model_fn)

# 定义训练的输入函数
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images':mnist.train.images},y=mnist.train.labels,
    batch_size=batch_size,num_epochs=None,shuffle=True)

model.train(input_fn,steps=num_steps)

# eval
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images':mnist.test.images},y=mnist.test.labels,
    batch_size=batch_size,num_epochs=None,shuffle=False
)

e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])

