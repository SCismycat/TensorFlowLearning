#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/23 9:12
# @Author  : Leslee
import tensorflow as tf
# 创建文件名队列
filename_queue = tf.train.string_input_producer(['stat0.csv','stat1.csv'])
# 创建读取csv文件的TextLineReader
reader = tf.TextLineReader()
# 从csv中取出一条数据记录value
_,value = reader.read(filename_queue)
_,multi_value = reader.read_up_to(filename_queue)
# 设置数据默认值
recoder_defaults = [[0],[0],[0.0],[0.0]]
# 使用decode_csv方法将数据记录转换为特征张量
id , age, income, outgo = tf.decode_csv(value,
                                        record_defaults=recoder_defaults)
features = tf.stack([id,age,income,outgo])
# 上述操作返回的是一个数据图上的操作，需要会话中执行才能生效。

