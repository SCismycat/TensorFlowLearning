#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.18 10:16

from __future__ import absolute_import,division,print_function

import gzip
import os
import re
import sys
import tarfile

import tensorflower.python.platform
from six.moves import urllib
import tensorflower as tf

from com.hiekn.tensorflow.CNN_TF_Tutorials import cifar10_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size',128,"Number of images to process in a batch")
tf.app.flags.DEFINE_string('data_dir', 'D:/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
# 全局变量
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.001
INITIAL_LEARNING_RATE = 0.01

# 如果使用多GPU，需要先指定tower_name作为所有op的前缀名字

TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def _activation_summary(x):
    """
    创建tensorboard的激活直方图
    :param x:
    :return:
    """
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME,'',x.op.name)
    tf.summary.histogram(tensor_name+'/activations',x)
    tf.summary.scalar(tensor_name+'/sparsity',tf.nn.zero_fraction(x))


def _variable_on_cpu(name,shape,initializer):
    # 保存参数到cpu
    with tf.device('/cpu:0'):
        var = tf.get_variable(name,shape,initializer=initializer)
    return var

def _variable_with_weight_decay(name,shape,stddev,wd):
    """
    帮助创建一个具有权重衰减的初始化变量。
    该变量用截断的正太分布初始化，仅当指定一个的时候，才添加权重衰减
    :param name:
    :param shape:
    :param stddev:
    :param wd:
    :return:
    """
    var = _variable_on_cpu(name,shape,tf.truncated_normal_initializer(stddev=stddev))

    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name="weight_loss")
        tf.add_to_collection('losses',weight_decay)
    return var

def distorted_inputs():
    # 使用reader op 为数据集构造一些变形的输入
    if not FLAGS.data_dir:
        raise ValueError('no data_dir')
    data_dir = os.path.join(FLAGS.data_dir,'cifar-10-batches-bin')
    return cifar10_input.distorted_inputs(data_dir=data_dir,
                                          batch_size=FLAGS.batch_size)

def inputs(eval_data):
    # 构造验证集输入
    if not FLAGS.data_dir:
        raise ValueError("please give a data dir")
    data_dir = os.path.join(FLAGS.data_dir,'cifar-10-batches-bin')
    return cifar10_input.inputs(eval_data=eval_data,data_dir=data_dir,
                                batch_size=FLAGS.batch_size)

def inference(images):

    # 构造模型
    # 卷积层1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[5,5,3,64],stddev=1e-4,wd=0.0)
        conv = tf.nn.conv2d(images,kernel,[1,1,1,1],padding='SAME')
        biases = _variable_on_cpu('biases',[64],tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name=scope.name)
        _activation_summary(conv1)
    # 池化层1
    pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name="pool1")
    # 正则化
    norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')

    # 卷积层2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[5,5,64,64],
                                             stddev=1e-4,wd=0.0)
        conv = tf.nn.conv2d(norm1,kernel,[1,1,1,1],padding='SAME')
        biases = _variable_on_cpu('biases',[64],tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(bias,name=scope.name)
        _activation_summary(conv2)
    # 正则化2
    norm2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm2')
    # 池化层2
    pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool2')


    # 线性修正的全连接层,拉平全连接层
    with tf.variable_scope('local3') as scope:
        dim = 1
        # 把 上一层输出的形状拉平
        for d in pool2.get_shape()[1:].as_list():
            dim *=d
        reshape = tf.reshape(pool2,[FLAGS.batch_size,dim])
        weights = _variable_with_weight_decay('weights',shape=[dim,384],
                                              stddev=0.04,wd=0.004)
        biases = _variable_on_cpu('biases',[384],tf.constant_initializer(0.1))

        local3 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope.name)
        _activation_summary(local3)
     # 线性修正的全连接层。
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights',shape=[384,192],
                                              stddev=0.04,wd=0.004)
        biases = _variable_on_cpu('biases',[192],tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3,weights)+biases,name=scope.name)
        _activation_summary(local4)

    # softmax层
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights',[192,NUM_CLASSES],
                                              stddev=1/192.0,wd=0.0)
        biases = _variable_on_cpu('biases',[NUM_CLASSES],tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4,weights),biases,name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear

def loss(logits,labels):
    # 对所有的可训练参数增加 l2 loss
    sparse_labels = tf.reshape(labels,[FLAGS.batch_size,1])
    indices = tf.reshape(tf.range(FLAGS.batch_size),[FLAGS.batch_size,1])
    concated = tf.concat([indices,sparse_labels],1)
    dense_labels = tf.sparse_to_dense(concated,[FLAGS.batch_size,NUM_CLASSES],
                                      1.0,0.0)
    # 计算均方误差
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits,dense_labels,name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'),name='total_loss')

def _add_loss_summaries(total_loss):

    loss_averages = tf.train.ExponentialMovingAverage(0.9,name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses+[total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name+' (raw)',l)
        tf.summary.scalar(l.op.name,loss_averages.average(l))

    return loss_averages_op


def train(total_loss,global_step):

    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
    tf.summary.scalar('learning_rate',lr)

    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name,var)
    for grad,var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name+'/gradients',grad)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op,variable_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op

def maybe_download_and_extract():
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory,filename)

    if not os.path.exists(filepath):
        def _progress(count,block_size,total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath,_ = urllib.request.urlretrieve(DATA_URL,filepath,reporthook=_progress)

        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)















































