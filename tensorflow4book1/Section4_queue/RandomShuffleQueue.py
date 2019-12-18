#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/18 17:19
# @Author  : Leslee

import tensorflow as tf

q = tf.RandomShuffleQueue(capacity=10,min_after_dequeue=2,dtypes="float")

sess = tf.Session()

for i in range(0,10):
    sess.run(q.enqueue(i))

# for i in range(0,8):
#     print(sess.run(q.dequeue()))

run_options = tf.RunOptions(timeout_in_ms = 10000)
try:
    print(sess.run(q.dequeue(),options=run_options))
except tf.errors.DeadlineExceededError:
    print('out of range')






