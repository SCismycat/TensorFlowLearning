#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Leslee
import  tensorflow as tf
from com.hiekn.tensorflow.Mnist_TF_work_1_2 import input_data

print("开始下载数据集..")
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print("下载完毕..")
sess = tf.InteractiveSession()
# 该函数可以更加灵活的构建代码，可以在运行计算的图的时候通过operation操作插入一些计算图。

x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10]) # 占位符

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10])) # 变量，跟占位符一样作为额外的输入量
sess.run(tf.initialize_all_variables())
y = tf.nn.softmax(tf.matmul(x,W)+b) # 使用softmax计算每个分类的概率

cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # 交叉熵

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # 训练使用最小梯度下降，且最小化交叉熵loss
init = tf.global_variables_initializer()
for i in range(1000):
    batch = mnist.train.next_batch(50)# load  mini-batchsize dataset
    train_step.run(feed_dict={x:batch[0],y_:batch[1]})
print("训练结束..")
"""
这段表达特别好：tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，
而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
"""
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuarcy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

print(accuarcy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))# 使用softmax取得效果有限

"""
开始使用CNN进行训练识别
"""
# 首先需要创建大量的W和b，由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# 定义卷积函数
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
# 定义最大池化函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
print("use cnn get feature..")
# 针对mnist开始卷积（第一层）
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

# 为了卷积需要，需要把x变成一个4d向量，2、3维度对应图片的宽、高。最后一个代表颜色通道数
x_image = tf.reshape(x,[-1,28,28,1])

# 然后，把x_image和权重张量进行卷积，加上偏置项，使用relu作为激活函数，最后进行max_pooling
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)

h_pool1 = max_pool_2x2(h_conv1)
print("second conv..")
# 第二层卷积
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层，图片尺寸减少到7*7，加入全连接层，就是一个全连接的神经网络，处理整张图片。
# 操作：这一步将池化层的tensor乘以W +b。计算relu
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# 输出层接softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])


y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# training and evaluate
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

sess.run(init)
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))