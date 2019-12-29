#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/27 11:21
# @Author  : Leslee
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print("开始下载数据集..")
mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
print("下载完毕..")

# 定义超参数
learning_rate = 0.01
max_train_steps = 1000
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
    ## 定义变量tf.Variable
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10])) # 变量，跟占位符一样作为额外的输入量
y = tf.nn.softmax(tf.matmul(x,W)+b) # 使用softmax计算每个分类的概率
# 定义损失函数
loss = -tf.reduce_sum(y_*tf.log(y)) # 交叉熵
# 创建优化器(优化器会专门学习一节)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# 定义单步训练操作
train_op = optimizer.minimize(loss)
# 上面两步可以写成一个步骤
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# 创建会话
with tf.Session() as sess:
    # 必须全局初始化
    init = tf.global_variables_initializer()
    sess.run(init)
    # 迭代训练
    for i in range(max_train_steps):
        batch = mnist.train.next_batch(batch_size)# load  mini-batchsize dataset
        train_op.run(feed_dict={x:batch[0],y_:batch[1]})
    print("训练结束..")
    # 评估模型
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuarcy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

    print(accuarcy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))# 使用softmax取得效果有限