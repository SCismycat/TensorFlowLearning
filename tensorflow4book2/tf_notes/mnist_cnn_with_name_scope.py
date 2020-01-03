#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/30 15:45
# @Author  : Leslee
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print("开始下载数据集..")
mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
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
    conv = tf.nn.conv2d(input,weights,strides=[1,1,1,1],padding='SAME')
    conv_1 = tf.nn.relu(conv+biases)
    max_pooling = tf.nn.max_pool(conv_1,
                                  ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
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
        return fullconnect_relu(input_data,[7*7*64,1024],[1024])

cnn_output = my_img_cnn_var_scope(x_image_input)
# 最后接全连接层。
hidden_pool_flat = tf.reshape(cnn_output,[-1,7*7*64])
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
    init = tf.global_variables_initializer()
    sess.run(init)
    # 迭代训练

    for i in range(max_train_steps):

        batch = mnist.train.next_batch(batch_size)# load  mini-batchsize dataset
        if i %100 == 0:
            train_acc = accuarcy.eval(feed_dict={
                x:batch[0],y_:batch[1],keep_prob:1.0})
            print("step %d, training accuracy %g" % (i, train_acc))
        train_op.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
    print("训练结束..")
    # 评估模型
    Saver.save(sess,"../model/")
    print(accuarcy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))# 使用softmax取得效果有限



