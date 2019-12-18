#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.27 14:05

import tensorflower as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager

# Import MNIST data
from tensorflower.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# Parameters
learning_rate = 0.001
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# 先分好数据的batchs
dataset = tf.data.Dataset.from_tensor_slices(
    (mnist.train.images,mnist.train.labels))
dateset = dataset.repeat().batch(batch_size).prefetch(batch_size)
dataset_iter = tfe.Iterator(dataset)

# 定义神经网络，使用eager API和tf.layer API
class NeuralNetwork(tfe.Network):
    def __init__(self):
        # 定义每个层
        super(NeuralNetwork, self).__init__()
        self.layer1 = self.track_layer(
            tf.layers.Dense(n_hidden_1,activation=tf.nn.relu))
        self.layer2 = self.track_layer(
            tf.layers.Dense(n_hidden_2,activation=tf.nn.relu))

        self.out_layer = self.track_layer(tf.layers.Dense(num_classes))

    def call(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.out_layer(x)

neural_network = NeuralNetwork()

# 交叉熵损失函数
def loss_fn(inference_fn,inputs,labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=inference_fn(inputs),labels=labels))

# 计算 acc
def accuracy_fn(inference_fn,inputs,labels):
    prediction = tf.nn.softmax(inference_fn(inputs))
    correct_pred = tf.equal(tf.argmax(prediction,1),labels)
    return tf.reduce_mean(tf.cast(correct_pred,tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate)

# 计算梯度
grad = tfe.implicit_gradients(loss_fn)

# 开始训练
average_loss = 0.
average_acc = 0.
for step in range(num_steps):
    d = dataset_iter.next()

    # Images
    x_batch = d[0]
    y_batch = tf.cast(d[1],dtype=tf.int64)

    # 计算整个batch的loss
    batch_loss = loss_fn(neural_network,x_batch,y_batch)
    average_loss += batch_loss
    # 计算整个batch的accuracy
    batch_accuracy = accuracy_fn(neural_network,x_batch,y_batch)
    average_acc += batch_accuracy
    if step == 0:
        # 打印优化前的初始的cost
        print("Initial loss= {:.9f}".format(average_loss))
    optimizer.apply_gradients(grad(neural_network,x_batch,y_batch))
    # 打印细节
    if (step+1) % display_step == 0 or step == 0:
        if step >0:
            average_loss/=display_step
            average_acc /= display_step
        print("Step:", '%04d' % (step + 1), " loss=",
              "{:.9f}".format(average_loss), " accuracy=",
              "{:.4f}".format(average_acc))
        average_loss = 0.
        average_acc = 0.

test_X = mnist.test.images
test_Y = mnist.test.labels

test_acc = accuracy_fn(neural_network,test_X,test_Y)
print("Testset Accuracy: {:.4f}".format(test_acc))




