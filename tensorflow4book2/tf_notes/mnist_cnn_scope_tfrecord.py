#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/31 15:16
# @Author  : Leslee
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

print("开始下载数据集..")
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
print("下载完毕..")

# 定义超参数
learning_rate = 0.01
max_train_steps = 20000
batch_size = 50
# .....
# 输入数据
print("开始下载数据集..")
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print("下载完毕..")

## 数据集转为tfrecord
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

## 读取MNIST数据集
images_train = mnist.train.images
labels_train = mnist.train.labels
pixels_train = images_train.shape[1]
num_examples_train = mnist.train.num_examples


image_test = mnist.test.images
label_test = mnist.test.labels
pixels_test = image_test.shape[1]
num_example_test = mnist.test.num_examples

#写入到tfrecord
def writer_to_tfrecord(path,pixels,labels,images,num_example):
    with tf.python_io.TFRecordWriter(path) as writer:
        for index in range(num_example):
            example = tf.train.Example(
                features = tf.train.Features(
                    feature= {
                        'pixels': _int64_feature(pixels),
                        'label': _int64_feature(np.argmax(labels[index])),
                        'image_raw': _byte_feature(images[index].tostring())
                    }
                )
            )
            writer.write(example.SerializeToString())
    print("write down..")
if not os.path.exists("./MNIST_data/train_mnist.tfrecord"):
    writer_to_tfrecord("./MNIST_data/train_mnist.tfrecord",pixels_train,
                   labels_train,images_train,num_examples_train)

    writer_to_tfrecord("./MNIST_data/test_mnist.tfrecord",pixels_test,
                   label_test,image_test,num_example_test)

def gen_tfrecord_data(input_data):
    filename_queue = tf.train.string_input_producer([input_data])
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
    images = tf.reshape(images, [28, 28, 1])
    images = tf.cast(images, tf.float32) * (1. / 255) - 0.5
    labels = tf.cast(features['label'],tf.int32)
    pixels = tf.cast(features['pixels'],tf.int32)
    return images,labels
## 获取数据
images_tr,labels_tr = gen_tfrecord_data("./MNIST_data/train_mnist.tfrecord")
images_te,labels_te = gen_tfrecord_data("./MNIST_data/test_mnist.tfrecord")
training_image_data = tf.image.resize_images(images_tr, [28, 28])
image_tr_batch,labels_tr_batch = tf.train.shuffle_batch([images_tr,labels_tr],
                                                        batch_size=50,
                                                        capacity=2000,
                                                        min_after_dequeue=1000)

# 构建模型
    ## 定义数据节点(占位符) tf.placeholder
x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10]) # 占位符
# 为cnn reshape
x_image_input = tf.reshape(x,[-1,28,28,1])
# 变量作用域
# 先定义一个CNN网络
def conv_relu(input,kernel_shape,bias_shape):
    # 创建或者获取名字是weights的变量
    weights = tf.get_variable("weights",kernel_shape,initializer=tf.truncated_normal_initializer(stddev=1.0))
    # 创建或获取名字是bias的变量
    biases = tf.get_variable("biases",bias_shape,initializer=tf.constant_initializer(0.1))
    conv = tf.nn.conv2d(input,weights,strides=[1,1,1,1],padding='VALID')
    conv_1 = tf.nn.relu(conv+biases)
    max_pooling = tf.nn.max_pool(conv_1,
                                  ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    return max_pooling
# 全连接层
def fullconnect_relu(input,weights_shape,bias_shape):
    fc_weights = tf.get_variable("fc_weights",weights_shape,initializer=tf.truncated_normal_initializer(stddev=1.0))
    biases = tf.get_variable("fc_biases",bias_shape,initializer=tf.constant_initializer(0.1))
    fc = tf.nn.relu(tf.matmul(input,fc_weights)+biases)
    return fc
# 输出层
def output_layers(input,weights_shape,bias_shape):
    op_weights = tf.get_variable("op_weights",weights_shape,initializer=tf.truncated_normal_initializer(stddev=1.0))
    op_biases = tf.get_variable("op_biases",bias_shape,initializer=tf.constant_initializer(0.1))
    return tf.nn.softmax(tf.matmul(input,op_weights)+op_biases)

def my_img_cnn_var_scope(input_img):
    with tf.variable_scope("conv1"):
        # 创建名为“conv1/weights”和conv1/biases的变量
        relu1 = conv_relu(input_img,[5,5,1,32],[32])
    with tf.variable_scope("conv2"):
        # 创建“conv1/weights”和conv1/biases的变量
        relu2 = conv_relu(relu1,[5,5,32,64],[64])
        return relu2

def my_full_connect_scope(input_data):
    with tf.variable_scope("fullconnect"):
        return fullconnect_relu(input_data,[50*4*4*64,1024],[1024])

cnn_output = my_img_cnn_var_scope(x_image_input)
# 最后接全连接层。
hidden_pool_flat = tf.reshape(cnn_output,[-1,50*4*4*64])
full_conc_output = my_full_connect_scope(hidden_pool_flat)
# 加dropout
keep_prob = tf.placeholder("float")
hidden_fullconnect_drop = tf.nn.dropout(full_conc_output,keep_prob=keep_prob)
y_conv = output_layers(hidden_fullconnect_drop,[1024,10],[10])

# 定义损失函数
loss = -tf.reduce_sum(y_*tf.log(y_conv)) # 交叉熵
# 创建优化器(优化器会专门学习一节)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
# 定义单步训练操作
train_op = optimizer.minimize(loss)
# 上面两步可以写成一个步骤
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
Saver = tf.train.Saver()
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuarcy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 创建会话
with tf.Session() as sess:
    # 必须全局初始化
    init_op = tf.group([tf.global_variables_initializer(),tf.local_variables_initializer()])
    sess.run(init_op)
    # 迭代训练
    # coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess)
    try:
        for i in range(max_train_steps):
            img_batch,lab_batch = sess.run([image_tr_batch,tf.one_hot(labels_tr_batch,depth=10)])
            if i %100 == 0:
                train_acc = accuarcy.eval(feed_dict={
                    x:img_batch,y_:lab_batch,keep_prob:1.0})
                print("step %d, training accuracy %g" % (i, train_acc))
            train_op.run(feed_dict={x:img_batch,y_:lab_batch,keep_prob:0.5})
    except tf.errors.OutOfRangeError:
        print("catch error")
    # finally:
    #     coord.request_stop()
    # coord.join(threads)
    print("训练结束..")
    # 评估模型
    Saver.save(sess,"./model/")
    print(accuarcy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))# 使用softmax取得效果有限
