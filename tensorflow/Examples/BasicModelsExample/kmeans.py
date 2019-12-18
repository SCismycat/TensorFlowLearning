#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.26 16:18

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
full_data_x = mnist.train.images

# 参数
num_steps = 50
batch_size = 1024
k = 25
num_classes = 10
num_features = 784

# 输入占位符
X = tf.placeholder(tf.float32,shape=[None,num_features])
# 标签-->用于分配质心标签进行测试
Y = tf.placeholder(tf.float32,shape=[num_classes])

# kmeans 参数
kmeans = KMeans(inputs=X,num_clusters=k,distance_metric='cosine',use_mini_batch=True)

# 构建 KMeans 的图
training_graph = kmeans.training_graph()

if len(training_graph) >6:
    (all_scores,cluster_idx,scores,cluster_centers_initialized,
     cluster_centers_var,init_op,train_op) = training_graph
else:
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     cluster_centers_var, init_op, train_op) = training_graph

cluster_idx = cluster_idx[0]
avg_distance = tf.reduce_mean(scores)

init_vars = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init_vars,feed_dict={X:full_data_x})
sess.run(init_op,feed_dict={X:full_data_x})

# training
for i in range(1,num_steps+1):
    _, d, idx = sess.run([train_op,avg_distance,cluster_idx],feed_dict={X:full_data_x})
    if i%10 ==0 or i ==1:
        print("Step %i, Avg Distance: %f" % (i, d))

# 为质心分配标签
# 使用每次训练的标签，汇总每个质心的所有标签总数
counts = np.zeros(shape=(k,num_classes))
for i in range(len(idx)):
    counts[idx[i]] += mnist.train.labels[i]

# 把最频繁的标签分配到质心
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)

# lookup：通过质心id映射到标签。
cluster_label = tf.nn.embedding_lookup(labels_map,cluster_idx)
# 计算acc
correct_prediction = tf.equal(cluster_label,tf.cast(tf.argmax(Y,1),tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 测试Model
test_x, test_y = mnist.test.images,mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))











