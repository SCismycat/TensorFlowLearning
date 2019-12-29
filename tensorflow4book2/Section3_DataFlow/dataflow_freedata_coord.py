#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/24 10:40
# @Author  : Leslee

import tensorflow as tf

filename_queue = tf.train.string_input_producer(['stat.tfrecord'],num_epochs=2)
# 创建一个reader
reader = tf.TextLineReader()

# 调用read选择一条数据
_,serialize = reader.read(filename_queue)

features = tf.parse_single_example(
    example_names=serialize,
    features={
        'id':tf.FixedLenFeature([],tf.int64),
        'age':tf.FixedLenFeature([],tf.int64),
        'income':tf.FixedLenFeature([],tf.float32),
        'outgo':tf.FixedLenFeature([],tf.float32)
    }
)
# 聚合两种初始化操作
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess = tf.Session()
sess.run(init_op)
# 创建协调器
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
# 打印程序后台线程信息
print('Threads:%s' %threads)
try:
    for i in range(10):
        if not coord.should_stop():
            example = sess.run(features)
            print(example)
except tf.errors.OutOfRangeError:
    print('Catch OutOfRangeError')
finally:
    # 请求停止所有后台进程
    coord.request_stop()
    print('Finished Reading')
# 等待后台进程退出
coord.join(threads)
sess.close()


"""
协调器工作原理：
监控TensorFlow所有后台进程，当出现异常线程的时候，should_stop变成
True。结束读取的For循环，然后使用request_stop方法，请求所有线程安全退出。
"""

# 批处理产生数据队列。

def get_many_example(filename_queue):
    reader = tf.SomeReader()
    _,value = reader.read(filename_queue)
    features = tf.decode_some(value)
    # 对样例进行预处理
    processed_example = 1#some_processing(features)
    return processed_example

def input_pipeline(filenames,batch_size,num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames,num_epochs=num_epochs,shuffle=True
    )
    example = get_many_example(filename_queue)

    # 出队样例个数
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3*batch_size
    example_batch = tf.train.shuffle_batch(
        [example],batch_size=batch_size,capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch


# 填充数据节点。feed_dict

x_batch = input_pipeline(['stat.tfrecord'],batch_size=20)
# 创建模型
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
sess.run(init_op)

# 使用coord来继续进行处理。














