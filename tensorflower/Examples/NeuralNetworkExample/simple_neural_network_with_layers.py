#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.27 13:32

""" Use TensorFlow 'layers' and 'estimator' API to build a simple neural network (a.k.a Multi-layer Perceptron) to classify MNIST digits dataset."""""
# 使用Tensorflow中的 layers图层和 estimator预测器的API构建简单的神经网络。
from tensorflower.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=False)

import tensorflower as tf

# Parameters
learning_rate = 0.1
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)


# 定义神经网络
def neural_network(x_dict):
    # 考虑到多输入情况，TF Estimator 输入是一个dict
    X = x_dict['images']
    # 第一层隐藏层
    layer_1 = tf.layers.dense(X,n_hidden_1)
    # 第二层隐藏层
    layer_2 = tf.layers.dense(layer_1,n_hidden_2)
    # 输出层
    out_layer = tf.layers.dense(layer_2,num_classes)
    return out_layer

# 定义模型函数(遵循TF estimitor 模板)
def model_fn(features,labels,mode):
    # 先构建神经网络
    logits = neural_network(features)

    # 预测结果
    pred_calsses = tf.argmax(logits,axis=1)
    pred_probas = tf.nn.softmax(logits)

    # 如果是 预测模式，早早的返回
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,predictions=pred_calsses)

    # 定义损失函数和优化函数
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,labels=tf.cast(labels,dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_op,global_step=tf.train.get_global_step())

    # 验证模型的acc
    acc_op = tf.metrics.accuracy(labels=labels,predictions=pred_calsses)

    # TF Estimators 需要返回一个EstimatorSpec，从而确定是(训练，预测)哪个操作
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_calsses,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy':acc_op})
    return estim_specs

# 简历Estimator
model = tf.estimator.Estimator(model_fn)

# 定义训练的输入函数
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images':mnist.train.images},y=mnist.train.labels,
    batch_size=batch_size,num_epochs=None,shuffle=True
)

# 训练模型
model.train(input_fn,steps=num_steps)

# 验证模型
# 定义验证的函数
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images':mnist.test.images},y=mnist.test.labels,
    batch_size=batch_size,shuffle=False
)
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])














