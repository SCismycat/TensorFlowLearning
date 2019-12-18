#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/17 16:10
# @Author  : Leslee

from __future__ import absolute_import,division,print_function

import os
import tensorflow.python.platform

import tensorflow as tf
from tensorflow.python.platform import gfile

IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    # 输入标准化
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key,value = reader.read(filename_queue)
    #
    record_bytes = tf.decode_raw(value,tf.uint8)
    # 把标签从int8转到int32
    result.label = tf.cast(tf.slice(record_bytes,[0],[label_bytes]),tf.int32)
    # reshape图片为长宽高的形式[depth,height,width]
    depth_major = tf.reshape(tf.slice(record_bytes,[label_bytes],[image_bytes]),
                             [result.depth,result.height,result.width])
    # 转化为[height,width,depth]
    result.uint8image = tf.transpose(depth_major,[1,2,0])

    return result

def _generate_image_and_label_batch(image,label,
                                    min_queue_examples,batch_size):
    """
    :param image: 3D Tensor
    :param label: 1D Tensor int32
    :param min_queue_examples:
    :param batch_size:
    :return:
    """
    num_preprocess_threads = 16
    image,label_batch = tf.train.shuffle_batch(
        [image,label],
        batch_size= batch_size,
        num_threads= num_preprocess_threads,
        capacity = min_queue_examples + 3*batch_size,
        min_after_dequeue=min_queue_examples)

    # 可视化训练数据
    tf.summary.image('images',image)
    return image,tf.reshape(label_batch,[batch_size])

def distorted_inputs(data_dir,batch_size):
    """
    :param data_dir: path to the cifar-10 data dict
    :param batch_size: number of image per batch
    :return:
        images: 4D tensor [batch_size,IMAGE_SIZE,IMAGE_SIZE,3]
        labels: 1D tensor [batch_size] size
    """
    filenames = [os.path.join(data_dir,'data_batch_%d.bin' % i) for i in range(1,6)]
    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file:' + f)

    # 产生要读取的文件名
    filename_queue = tf.train.string_input_producer(filenames)

    # read examples from files in the filename queue
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image,tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 对图像进行随机变形，产生训练的数据

    # 随机裁剪
    distorted_image = tf.image.random_crop(reshaped_image,[height,width,3])
    # 随机翻转
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)

    distorted_image = tf.image.random_contrast(distorted_image,lower=0.2,upper=1.8)

    float_image = tf.image.per_image_standardization(distorted_image)

    min_fraction_of_examples_in_queue = 0.4

    min_queue_examples = int(NUM_EXAMPLES_PER_FOR_TRAIN*
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)
    return _generate_image_and_label_batch(float_image,read_input.label,
                                           min_queue_examples,batch_size)

def inputs(eval_data,data_dir,batch_size):
    if not eval_data:
        filenames = [os.path.join(data_dir,'data_batch_%d.bin' % i) for i in range(1,6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir,"test_batch.bin")]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('failed to find file:'+f)

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image,tf.float32)
    height = IMAGE_SIZE
    width = IMAGE_SIZE

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,width,height)

    float_image = tf.image.per_image_standardization(reshaped_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch*min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image,read_input.label,min_queue_examples,batch_size)











































