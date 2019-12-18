#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.27 17:17
# 不定长的序列处理方式
import tensorflower as tf
import random

# 数据生成
class ToySequenceData(object):
    def __init__(self,n_samples=1000,max_seq_len=20,min_seq_len=3,
                 max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            len = random.randint(min_seq_len,max_seq_len)
            self.seqlen.append(len) # 存储每个序列的长度，方便动态计算

            if random.random() < .5:
                # 生成一个线性序列
                rand_start = random.randint(0,max_value - len)
                s = [[float(i)/max_value] for i in
                     range(rand_start+rand_start+len)]
                # 填充序列，满足一致性
                s += [[0.] for i in range(max_seq_len-len)]
                self.data.append(s)
                self.labels.append([1.,0.])
            else:
                s = [[float(random.randint(0,max_value))/max_value]
                     for i in range(len)]
                # 填充序列
                s += [[0.] for i in range(max_seq_len-len)]
                self.data.append(s)
                self.labels.append([0.,1.])
        self.batch_id = 0

    def next(self,batch_size):

        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id+
                                                  batch_size,len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id+
                                                      batch_size,len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id+
                                                      batch_size,len(self.data))])
        self.batch_id = min(self.batch_id+batch_size,len(self.data))

        return  batch_data,batch_labels,batch_seqlen


# Model

learning_rate = 0.01
training_steps = 10000
batch_size = 128
display_step = 200

seq_max_len = 20
n_hidden = 64
n_classes = 2

trainset = ToySequenceData(n_samples=1000,max_seq_len=seq_max_len)
testset = ToySequenceData(n_samples=500,max_seq_len=seq_max_len)

x = tf.placeholder("float",[None,seq_max_len,1])
y = tf.placeholder("float",[None,n_classes])
seqlen = tf.placeholder(tf.int32,[None])

# 定义权重与bias
weights = {
    'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))
}
biases = {
    'out':tf.Variable(tf.random_normal([n_classes]))
}

def dynamicRNN(x,seqlen,weights,biases):

    x = tf.unstack(x,seq_max_len,1)

    # 定义lstm cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell,x,dtype=tf.float32,
                                                sequence_length=seqlen)
    # 执行动态计算的时候，必须检索最后一个动态计算的输出，如果序列长度为10 ，需要检索第10个输出。
    # 所以自定义一个OP，针对每个样本的batchsize，获取其长度并且获得相应的输出。
    # outputs 是每个timesteps的输出列表，打包成[batch_size,n_step,n_inputs]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs,[1,0,2])

    batch_size = tf.shape(outputs)[0]
    # 每个样本的起始索引
    index = tf.range(0,batch_size) * seq_max_len + (seqlen-1)

    outputs = tf.gather(tf.reshape(outputs,[-1,n_hidden]),index)

    return tf.matmul(outputs,weights['out']) + biases['out']

pred = dynamicRNN(x,seqlen,weights,biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init  = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1,training_steps+1):
        batch_x, batch_y , batch_seqlen = trainset.next(batch_size)

        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,
                                      seqlen:batch_seqlen})
        if step % display_step == 0 or step ==1:
            acc,loss = sess.run([accuracy,cost],feed_dict={x:batch_x,y:batch_y,seqlen:batch_seqlen})
            print("Step " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
    print("Optimization Finished!")

    # Calculate accuracy
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                        seqlen: test_seqlen}))

