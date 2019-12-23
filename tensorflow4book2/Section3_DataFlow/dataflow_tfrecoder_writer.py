#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/23 9:21
# @Author  : Leslee
import tensorflow as tf

# 创建向TFRecoder文件写入数据记录的writer
writer = tf.python_io.TFRecordWriter('stat.tfrecord')
# 构造输入样例
for i in range(1,3):
    # 创建example.proto定义的样例
    example = tf.train.Example(
        features = tf.train.Features(
            feature = {
                'id':tf.train.Feature(int64_list=
                                      tf.train.Int64List(value=[i])),
                'age':tf.train.Feature(int64_list=
                                      tf.train.Int64List(value=[i*24])),
                'income':tf.train.Feature(float_list=
                                          tf.train.FloatList(value=[i*2048.0])),
                'outgo':tf.train.Feature(float_list=
                                         tf.train.FloatList(value=[i*1024.0]))
            }
        )
    )
    # 序列化为字符串，写入到 stat.tfrecord文件
    writer.write(example.SerializeToString())
writer.close()
