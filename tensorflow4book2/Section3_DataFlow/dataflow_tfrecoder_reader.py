#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/23 9:31
# @Author  : Leslee

import tensorflow as tf

# 创建文件名队列filename_queue
filename_queue = tf.train.string_input_producer(['stat.tfrecord'])
# 创建读取TFReocrd读取reader
reader = tf.TFRecordReader()

# 取出stat.tfrecord文件中一条序列化样例
_,serialized_example = reader.read(filename_queue)
# 将一条序列化样例转换为其包含的所有特征张量
features = tf.parse_single_example(
    serialized_example,
    features={
        'id':tf.FixedLenFeature([],tf.int64),
        'age':tf.FixedLenFeature([],tf.int64),
        'income':tf.FixedLenFeature([],tf.float32),
        'outgo':tf.FixedLenFeature([],tf.float32)
    }
)
print(features)

init_read_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_read_op)
# 启动执行入队操作的后台线程
tf.train.start_queue_runners(sess=sess)









