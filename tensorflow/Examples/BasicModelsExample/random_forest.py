#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.26 16:45

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# parameters
num_steps = 500 # 训练总步数
batch_size = 1024
num_classes = 10
num_features = 784
num_trees = 10
max_nodes = 1000

X = tf.placeholder(tf.float32,shape=[None,num_features])
Y = tf.placeholder(tf.float32,shape=[None])

hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()
# 构建随机森林
forgest_graph = tensor_forest.RandomForestGraphs(hparams)
# 获取训练图和损失
train_op = forgest_graph.training_graph(X,Y)
loss_op = forgest_graph.training_loss(X,Y)

# 衡量准确率
infer_op,_, _ = forgest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op,1),tf.cast(Y,tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init_vars = tf.group(tf.global_variables_initializer(),
                     resources.initialize_resources(resources.shared_resources()))

sess = tf.Session()

sess.run(init_vars)

for i in range(1,num_steps+1):
    batch_x,batch_y = mnist.train.next_batch(batch_size)
    _, l = sess.run([train_op,loss_op],feed_dict={X:batch_x,Y:batch_y})
    if i % 50 == 0 or i ==1:
        acc = sess.run(accuracy_op,feed_dict={X:batch_x,Y:batch_y})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

test_x,test_y = mnist.test.images,mnist.test.labels
print("Test Acc:",sess.run(accuracy_op,feed_dict={X:test_x,Y:test_y}))































