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
    ## 定义变量tf.Variable
W_conv1 = tf.Variable(initial_value=tf.truncated_normal(shape=[5,5,1,32],stddev=0.1))
b_conv1 = tf.Variable(initial_value=tf.constant(0.1,shape=[32]))

x_image_input = tf.reshape(x,[-1,28,28,1])
# 卷积层
hidden_conv1 = tf.nn.conv2d(x_image_input,W_conv1,strides=[1,1,1,1],padding='SAME')
hidden_conv1_active1 = tf.nn.relu(hidden_conv1+ b_conv1)
# 池化层
hidden_pool1 = tf.nn.max_pool(hidden_conv1_active1,
                              ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# 再来一层CNN
W_conv2 = tf.Variable(initial_value=tf.truncated_normal(shape=[5,5,32,64],stddev=0.1))
b_conv2 = tf.Variable(initial_value=tf.constant(0.1,shape=[64]))

hidden_conv2 = tf.nn.conv2d(hidden_pool1,W_conv2,strides=[1,1,1,1],
                            padding='SAME')
hidden_conv2_active = tf.nn.relu(hidden_conv2+b_conv2)
# 池化
hidden_pool2 = tf.nn.max_pool(hidden_conv2_active,
                              ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 最后接全连接层。
W_fullconnect = tf.Variable(initial_value=tf.truncated_normal(shape=[7*7*64,1024],stddev=0.1))
b_fullconnect = tf.Variable(initial_value=tf.constant(0.1,shape=[1024]))

hidden_pool_flat = tf.reshape(hidden_pool2,[-1,7*7*64])
hidden_fullconnect = tf.nn.relu(tf.matmul(hidden_pool_flat,W_fullconnect)+b_fullconnect)

# 加dropout
keep_prob = tf.placeholder("float")
hidden_fullconnect_drop = tf.nn.dropout(hidden_fullconnect,keep_prob=keep_prob)

# 输出层softmax
w_softmax = tf.Variable(initial_value=tf.truncated_normal(shape=[1024,10],stddev=0.1))
b_softmax = tf.Variable(initial_value=tf.constant(0.1,shape=[10]))

y_conv = tf.nn.softmax(tf.matmul(hidden_fullconnect_drop,w_softmax)+b_softmax)

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

def my_img_cnn(input_image):
    # 第一层
    conv1_w = tf.Variable(tf.random_normal([5,5,32,32]),name='conv1_w')
    conv1_b = tf.Variable(tf.constant(0.1,shape=[32]),name='conv1_b')
    conv1 = tf.nn.conv2d(input_image,conv1_w,strides=[1,1,1,1],padding='SAME')
    conv1_active = tf.nn.relu(conv1+conv1_b)
    # 第二层
    conv2_w = tf.Variable(tf.random_normal([5,5,32,32]),name='conv2_w')
    conv2_b = tf.Variable(tf.constant(0.1,shape=[32]),name='conv2_b')
    conv2 = tf.nn.conv2d(conv1_active,conv2_w,strides=[1,1,1,1],padding='SAME')
    conv2_active = tf.nn.relu(conv2+conv2_b)
    return conv2_active
# 变量作用域
# 先定义一个CNN网络
def conv_relu(input,kernel_shape,bias_shape):
    # 创建或者获取名字是weights的变量
    weights = tf.get_variable("weights",kernel_shape,initializer=tf.random_normal_initializer())
    # 创建或获取名字是bias的变量
    biases = tf.get_variable("weights",kernel_shape,initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input,weights,strides=[1,1,1,1],padding='SAME')
    return tf.nn.relu(conv+biases)
def my_img_cnn_var_scope(input_img):
    with tf.variable_scope("conv1"):
        # 创建名为“conv1/weights”和conv1/biases的变量
        relu1 = conv_relu(input_img,[5,5,32,32],[32])
    with tf.variable_scope("conv2"):
        # 创建“conv1/weights”和conv1/biases的变量
        relu2 = conv_relu(relu1,[5,5,32,32],[32])
        return relu2


