#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/31 14:45
# @Author  : Leslee
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# 定义函数转化变量类型。
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 将数据转化为tf.train.Example格式。
def _make_example(pixels, label, image):
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(label)),
        'image_raw': _bytes_feature(image_raw)
    }))
    return example


# 读取mnist训练数据。
mnist = input_data.read_data_sets("../MNIST_data", dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

with tf.python_io.TFRecordWriter("../MNIST_data/output_train.tfrecord") as writer:
    for idx in range(num_examples):
        example = _make_example(pixels,labels[idx],images[idx])
        writer.write(example.SerializeToString())

filename_queue = tf.train.string_input_producer(['../MNIST_data/output_train.tfrecord'])
reader = tf.TFRecordReader()
_,values = reader.read(filename_queue)

features = tf.parse_single_example(
    values,
    features={
        'pixels': tf.FixedLenFeature([],tf.int64),
        'label': tf.FixedLenFeature([],tf.int64),
        'image_raw':tf.FixedLenFeature([],tf.string)
    }
)
images = tf.decode_raw(features['image_raw'],tf.uint8)
labels = tf.cast(features['label'],tf.int32)
pixels = tf.cast(features['pixels'],tf.int32)

sess = tf.Session()
coord = tf.train.Coordinator()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess,coord=coord)
for i in range(2):
    example = sess.run(features)
    print(example)
sess.close()

# 读取mnist测试数据。
# images_test = mnist.test.images
# labels_test = mnist.test.labels
# pixels_test = images_test.shape[1]
# num_examples_test = mnist.test.num_examples
#
# # 输出包含测试数据的TFRecord文件。
# with tf.python_io.TFRecordWriter("../MNIST_TF/output_test.tfrecords") as writer:
#     for index in range(num_examples_test):
#         example = _make_example(
#             pixels_test, labels_test[index], images_test[index])
#         writer.write(example.SerializeToString())
# print("TFRecord测试文件已保存。")


