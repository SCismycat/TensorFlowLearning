#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/24 11:50
# @Author  : Leslee
"""
cifar-10数据集最佳实践
1. 创建文件名列表，直接读取。
2. 创建文件名队列：用tf.train.string_input_producer
3. 创建reader和decoder，在decoder把图片reshape成模型输入的shape
4. 创建样例队列

"""
import tensorflow as tf

LABEL_BYTES = 1
IMAGE_SIZE = 32
IMAGE_IMAGE_DEPTH = 3

IMAGE_BYTES = IMAGE_SIZE * IMAGE_SIZE * IMAGE_IMAGE_DEPTH
NUM_CLASSES = 10

def read_cifar10(data_file,batch_size):
    """
    :param data_file: 数据文件
    :param batch_size:
    :return:
        images:[batch_size,image_size,image_size,3] 图像批数据
        labels:[batch_size,NUM_CLASSES]
    """
    # 单挑数据字节数
    record_bytes = LABEL_BYTES + IMAGE_BYTES
    # 创建文件名列表
    data_files = tf.gfile.Glob(data_file)
    # 创建文件名队列
    file_queue = tf.train.string_input_producer(data_files,shuffle=True)

    # 创建二进制文件对应的Reader
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _,value = reader.read(file_queue)
    # 将样例分拆为列表标签和图片
    record = tf.reshape(tf.decode_raw(value,tf.uint8),[record_bytes])
    label = tf.cast(tf.slice(record,[0],[LABEL_BYTES]),tf.int32)
    # 把[depth*height*width]转为[depth,height,depth]的图片Tensor
    depth_major = tf.reshape(tf.slice(record,[LABEL_BYTES],[IMAGE_BYTES]),
                             [IMAGE_IMAGE_DEPTH,IMAGE_SIZE,IMAGE_SIZE])
    # shape
    image = tf.cast(tf.transpose(depth_major,[1,2,0]),tf.float32)

    # 创建样例队列，初始化一个随机队列，指定好参数
    example_queue = tf.RandomShuffleQueue(
        capacity=16*batch_size,
        min_after_dequeue=8*batch_size,
        dtypes=[tf.float32,tf.int32],
        shapes=[[IMAGE_SIZE,IMAGE_SIZE,IMAGE_IMAGE_DEPTH],[1]])
    num_threads = 16
    # 创建样例队列的入队操作
    example_enqueue_op = example_queue.enqueue([image,label])
    # 把定义的16个线程全部加入到queue runner
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue,[example_enqueue_op] *num_threads))
    # 按照batch_size大小出队
    images, labels = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels,[batch_size,1])
    indices = tf.reshape(tf.range(0,batch_size,1),[batch_size,1])
    labels = tf.sparse_to_dense(
        tf.concat(values=[indices,labels],axis=1),
        [batch_size,NUM_CLASSES],1.0,0.0)
    return images,labels

data_file = "E:/PythonProject/data/tflearn/cifar-10-batches-py/data_batch_*"
img,lab = read_cifar10(data_file,32)
print(img,lab)


