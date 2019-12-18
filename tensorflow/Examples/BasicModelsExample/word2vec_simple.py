#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.26 17:43

import collections
import os
import random
import urllib.request
import zipfile

import numpy as np
import tensorflow as tf

# 训练参数
learning_rate = 0.1
batch_size = 128
num_steps = 3000000
display_step = 10000
eval_step = 200000

# 验证参数
eval_words = [b'five', b'of', b'going', b'hardware', b'american', b'britain']

# word2vec 参数
embedding_size = 200
max_vocab_size = 50000
min_occurrence = 10
skip_window = 3
num_skips = 2
num_sampled = 64

url = 'http://mattmahoney.net/dc/text8.zip'
data_path = 'text8.zip'

if not os.path.exists(data_path):
    print("download dataset")
    filename, _ = urllib.request.urlretrieve(url,data_path)
    print("Done!")

with zipfile.ZipFile(data_path) as f:
    text_words = f.read(f.namelist()[0]).lower().split()

count = [('UNK',-1)]

count.extend(collections.Counter(text_words).most_common(max_vocab_size-1))
# 删除掉最小共现次数的词
for i in range(len(count)-1,-1,-1):
    if count[i][1] <min_occurrence:
        count.pop(i)
    else:
        break

vocab_size = len(count)

# 为单词分配ID
word2id = dict()
for i, (word,_) in enumerate(count):
    word2id[word] = i

data = list()
unk_count = 0
for word in text_words:
    index = word2id.get(word,0)
    if index == 0:
        unk_count += 1
    data.append(index)

count[0] = ('UNK',unk_count)
id2word = dict(zip(word2id.values(),word2id.keys()))

print("Words count:", len(text_words))
print("Unique words:", len(set(text_words)))
print("Vocabulary size:", vocab_size)
print("Most common words:", count[:10])

data_index = 0
#为skip-gram model生成训练数据

def next_batch(batch_size,num_skips,skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <=2 *skip_window
    batch = np.ndarray(shape=(batch_size),dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)
    # 获取窗口尺寸(左右的词 + 当前词)
    span = 2*skip_window +1
    buffer = collections.deque(maxlen=span)

    if data_index + span >len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index+span])
    data_index += span
    for i in range(batch_size//num_skips):
        context_words = [w for w in range(span) if w!=skip_window]
        words_to_use = random.sample(context_words,num_skips)
        for j ,context_words in enumerate(words_to_use):
            batch[i*num_skips+j] = buffer[skip_window]
            labels[i*num_skips+j,0] = buffer[context_words]
        if data_index ==len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index +=1
    data_index = (data_index +len(data)-span) % len(data)
    return batch,labels

# Input data
X = tf.placeholder(tf.int32,shape=[None])
Y = tf.placeholder(tf.int32,shape=[None,1])

with tf.device("/cpu:0"):
    embedding = tf.Variable(tf.random_normal([vocab_size,embedding_size]))
    X_embed = tf.nn.embedding_lookup(embedding,X)

    # 为NCE loss 构造变量
    nce_weights = tf.Variable(tf.random_normal([vocab_size,embedding_size]))
    nce_biases = tf.Variable(tf.zeros([vocab_size]))

# 为每个batch计算平均的nce loss
loss_op = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=Y,
                   inputs=X_embed,
                   num_sampled=num_sampled,
                   num_classes=vocab_size))

# 定义优化器，优化器的作用是求导，反向传播。
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)

# 验证/评价
# 计算输入数据embedding 和每个embedding向量的cosine相似度
X_embed_norm = X_embed / tf.sqrt(tf.reduce_sum(tf.square(X_embed)))
embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding),1,keep_dims=True))
cosine_sim_op = tf.matmul(X_embed_norm,embedding_norm,transpose_b=True)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    x_test = np.array([word2id[w] for w in eval_words])

    average_loss = 0

    for step in range(1,num_steps+1):
        batch_x, batch_y = next_batch(batch_size,num_skips,skip_window)
        # 训练
        _,loss = sess.run([train_op,loss_op],feed_dict={X:batch_x,Y:batch_y})
        average_loss += loss

        if step % display_step ==0 or step == 1:
            if step >1:
                average_loss /= display_step
            print("Step "+str(step)+",Average loss= "+"{:.4f}".format(average_loss))
            average_loss = 0

        # evaluation
        if step % display_step == 0 or step == 1:
            print("evaluation..")
            sim = sess.run(cosine_sim_op,feed_dict={X:x_test})
            for i in range(len(eval_words)):
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = '"%s" nearest neighbors:' % eval_words[i]
                for k in range(top_k):
                    log_str = '%s %s,' % (log_str, id2word[nearest[k]])
                print(log_str)











