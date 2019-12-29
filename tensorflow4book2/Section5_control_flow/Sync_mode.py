#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/25 10:56
# @Author  : Leslee
import tensorflow as tf
FLAGS = tf.app.flags
hid = 0
W,b = 0.0,0.0
y_ = 1
y = tf.nn.softmax(tf.nn.xw_plus_b(hid,W,b))
num_workers = 10
global_step = 100

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))

opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
# FLAGS.sync_replicas为True，表示使用同步训练机制更新参数。
if FLAGS.sync_replicas:
    if FLAGS.replicas_to_aggregate is None:
        replicas_to_aggregate = num_workers
    else:
        replicas_to_aggregate = FLAGS.replicas_to_aggregate
    # 创建同步优化器
    opt = tf.train.SyncReplicasOptimizer(
        opt,
        replicas_to_aggregate=replicas_to_aggregate,
        total_num_replicas=num_workers,
        name='mnist_sync_replicas')
train_step = opt.minimize(cross_entropy,global_step=global_step)







