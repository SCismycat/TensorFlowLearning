#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/18 17:29
# @Author  : Leslee
# 队列管理器
import tensorflow as tf

q = tf.FIFOQueue(1000,"float")
counter = tf.Variable(0.0) # 计数器
increment_op = tf.assign_add(counter,tf.constant(1.0)) # 给计数器 +1
enqueue_op = q.enqueue(counter) # 计数器的值加入到队列中
# 队列管理器
qr = tf.train.QueueRunner(q,enqueue_ops=[increment_op,enqueue_op] * 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()

    enqueue_threads = qr.create_threads(sess,coord=coord,start=True)

    # 主线程
    for i in range(10):
        print(sess.run(q.dequeue()))
    coord.request_stop()  # 通知其他线程关闭
    coord.join(enqueue_threads)
# queueruner 存在一个问题： 入队线程自顾自的执行，在需要出队操作完成之后，程序没法结束。

