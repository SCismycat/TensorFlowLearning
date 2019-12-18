#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.18 17:37

from __future__ import absolute_import,division,print_function

import argparse,collections,hashlib,math,os,random,sys,zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

from tempfile import gettempdir
from tensorflow.contrib.tensorboard.plugins import projector


data_index = 0

def _hash_file(fpath):
    hasher = hashlib.sha256()
    with open(fpath,'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(65535),b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def word2vec_basic(log_dir):
    # 创建tensorboard的可视化目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 第一步，下载数据
    url = 'http://mattmahoney.net/dc/'

    def maybe_download(filename,expected_bytes,sha256=None):
        local_filename = os.path.join(gettempdir(),filename)
        if not os.path.exists(local_filename):
            local_filename,_ = urllib.request.urlretrieve(url+filename,
                                                          local_filename)
        statinfo = os.stat(local_filename)

        if sha256 and _hash_file(local_filename) !=sha256:
            raise Exception('Failed to verify ' + local_filename + ' due to hash '
                           'mismatch. Can you get to it with a browser?')

        if statinfo.st_size == expected_bytes:
            print("found and verified",filename)
        else:
            print(statinfo.st_size)
            raise Exception('Failed to verify ' + local_filename +
                            '. Can you get to it with a browser?')
        return local_filename

    filename = maybe_download('text8.zip',31344016,
      sha256='a6640522afe85d1963ad56c05b0ede0a0c000dddc9671758a6cc09b7a38e5232')

    # 数据转为List<String>
    def read_data(filename):
        with zipfile.ZipFile(filename) as f:
            data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        return data

    vocabulary = read_data(filename)
    print('data_size',len(vocabulary))

    # 第二步，建词典并且把罕见词替换成UNK
    vocabulary_size = 50000

    def build_dataset(words,n_words):

        count = [['UNK',-1]]
        count.extend(collections.Counter(words).most_common(n_words-1))
        dictionary = {word:index for index,(word,_) in enumerate(count)}
        data = []
        unk_count = 0
        for word in words:
            index = dictionary.get(word,0)
            if index ==0: # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
        return data,count,dictionary,reversed_dictionary

    # data: 词表中的所有的词的id
    # count： 单词和出现次数的map
    # dictionary： 单词-->index 的映射
    # reverse_dictionary:index -->单词
    data,count,dictionary,reversed_dictionary = build_dataset(vocabulary,vocabulary_size)
    del vocabulary
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reversed_dictionary[i] for i in data[:10]])

    # 针对skip-gram模型生成batch数据
    def generate_batch(batch_size,num_skips,skip_window):
        global data_index
        assert batch_size % num_skips ==0
        assert num_skips <= 2*skip_window
        batch = np.ndarray(shape=(batch_size),dtype=np.int32)
        labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)
        # skip的范围
        span = 2 * skip_window + 1
        buffer = collections.deque(maxlen=span)
        if data_index + span > len(data):
            data_index = 0
        buffer.extend(data[data_index:data_index+span]) # 向后取一个窗口内的结果
        data_index += span
        for i in range(batch_size//num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words,num_skips)
            for j,context_words in enumerate(words_to_use):
                batch[i*num_skips +j] = buffer[skip_window]
                labels[i*num_skips+j,0] = buffer[context_words]
            if data_index == len(data):
                buffer.extend(data[0:span])
                data_index = span
            else:
                buffer.append(data[data_index])
                data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        data_index = (data_index-span) % len(data)
        return batch,labels

    batch,labels = generate_batch(batch_size=8,num_skips=2,skip_window=1)
    for i in range(8):
        print(batch[i],reversed_dictionary[batch[i]], '->', labels[i, 0],
          reversed_dictionary[labels[i, 0]])

    # 建立并且训练模型

    batch_size = 128
    embedding_size = 128 # 词向量维度
    skip_window = 1 # 考虑左右几个单词
    num_skips = 2 # 复用输入生成标签的次数
    num_sampled = 64 # 负样本数量

    # 采样一个样本的近邻作为随机验证机，将验证集样本限制为 较低id的单词，是比较高频的构造词汇
    # 这三个变量用作显示模型准确率，不影响计算。
    valid_size = 16 # 用于评估相似性的随机单词集合
    valid_window = 100 #
    valid_examples = np.random.choice(valid_window,valid_size,replace=False)

    graph = tf.Graph()

    with graph.as_default():

        # 输入数据
        with tf.name_scope('input'):
            train_inputs = tf.placeholder(tf.int32,shape=[batch_size])
            train_labels = tf.placeholder(tf.int32,shape=[batch_size,1])
            valid_dataset = tf.constant(valid_examples,dtype=tf.int32)

        # 操作op和变量variables 固定在CPU上。
        with tf.device('/cpu:0'):
            with tf.name_scope('embeddings'):
                embeddings = tf.Variable(
                    tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
                embed = tf.nn.embedding_lookup(embeddings,train_inputs)

            # 构造NCE损失的变量
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(
                    tf.truncated_normal([vocabulary_size,embedding_size],stddev=1.0/math.sqrt(embedding_size)))

            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # 计算该批次的平均nce损失，当评估损失的时候，自动绘制一个新的负样本。
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights= nce_weights,
                    biases= nce_biases,
                    labels = train_labels,
                    inputs=embed,
                    num_sampled=num_sampled,
                    num_classes= vocabulary_size))
        # 汇总损失
        tf.summary.scalar('loss',loss)

        # 构造SGD
        with tf.name_scope('opytimizer'):
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        # 计算小批次样本和所有样本之间的余弦相似度
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
        normalized_embeddings = embeddings/norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
        similarity = tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)

        # merge all summary
        merged = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

    # 开始训练
    num_steps = 1000001

    with tf.compat.v1.Session(graph=graph) as session:
        # 写入摘要
        writer = tf.summary.FileWriter(log_dir,session.graph)

        init.run()
        print('inited..')
        average_loss = 0
        for step in range(num_steps):
            batch_inputs,batch_labels = generate_batch(batch_size,num_skips,skip_window)
            feed_dict = {train_inputs:batch_inputs,train_labels:batch_labels}
            # 定义元变量
            run_metadata = tf.RunMetadata()

            _,summary,loss_val = session.run([optimizer,merged,loss],
                                             feed_dict=feed_dict,
                                             run_metadata=run_metadata)
            average_loss += loss_val


            writer.add_summary(summary,step)

            if step == (num_steps -1):
                writer.add_run_metadata(run_metadata,'step%d' % step)

            if step % 2000 ==0:
                if step>0:
                    average_loss /=2000
                    # 平均损失是对最近的2000个批次样本的估计。
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reversed_dictionary[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word

                    print(log_str,', '.join([reversed_dictionary[nearest[k]] for k in range(top_k)]))
        final_embeddings = normalized_embeddings.eval()

        # 写下embedding的相应标签
        with open(log_dir+'/metadata.tsv','w') as f:
            for i in range(vocabulary_size):
                f.write(reversed_dictionary[i] +'\n')

        # 保存checkpoint
        saver.save(session,os.path.join(log_dir,'model.ckpt'))

        # 配置Tensorboard
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = os.path.join(log_dir,'metadata.tsv')
        projector.visualize_embeddings(writer,config)
    writer.close()

    # Step 6: Visualize the embeddings.

    # pylint: disable=missing-docstring
    # Function to draw visualization of distance between embeddings.
    def plot_with_labels(low_dim_embs, labels, filename):
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom')

        plt.savefig(filename)

    try:
        # pylint: disable=g-import-not-at-top
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        tsne = TSNE(
            perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reversed_dictionary[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(),
                                                            'tsne.png'))

    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)


# All functionality is run after tf.compat.v1.app.run() (b/122547914). This
# could be split up but the methods are laid sequentially with their usage for
# clarity.
def main(unused_argv):
    # Give a folder path as an argument with '--log_dir' to save
    # TensorBoard summaries. Default is a log folder in current directory.
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(current_path, 'log'),
        help='The log directory for TensorBoard summaries.')
    flags, unused_flags = parser.parse_known_args()
    word2vec_basic(flags.log_dir)


if __name__ == '__main__':
    tf.app.run()



































