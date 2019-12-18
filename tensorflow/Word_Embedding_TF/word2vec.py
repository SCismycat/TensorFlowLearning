#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.20 15:00

from __future__ import absolute_import,division

import os,sys,threading,time
import tensorflow.python.platform

from six.moves import xrange
import numpy as np
import tensorflow as tf
from com.hiekn.tensorflow.Word_Embedding_TF import gen_word2vec as word2vec


flags = tf.compat.v1.app.flags
# 保存训练摘要和模型
flags.DEFINE_string("save_path", None, "Directory to write the model and "
                    "training summaries.")
# 训练数据，纯文本
flags.DEFINE_string("train_data", None, "Training text file. "
                    "E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
# 验证数据集，是一个类比数据集 A = B-C+D
flags.DEFINE_string("eval_data", None, "File consisting of analogies of four tokens."
    "embedding 2 - embedding 1 + embedding 3 should be close "
    "to embedding 4."
    "E.g. https://word2vec.googlecode.com/svn/trunk/questions-words.txt.")
# 词向量维度
flags.DEFINE_integer("embedding_size",200,"embedding dimession size")
# 训练的epoch
flags.DEFINE_integer("epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
# 学习率
flags.DEFINE_float("learning rate",0.2,"Initial learning rate.")
# 每次训练的负采样样本数
flags.DEFINE_integer("num_neg_samples", 100,
                     "Negative samples per training example.")
#batch_size
flags.DEFINE_integer("batch_size", 16,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
# 并行训练的数量
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
# skip-gram的窗口大小。即目标词预测左右词汇的词数
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
# 词汇表中最小单词出现次数
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
# 出现频率较高的词，被随机下采样。
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
# 设置IPython
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy('france', 'paris', 'russia') and "
    "model.nearby(['proton', 'elephant', 'maxwell']")

flags.DEFINE_integer("statistics_interval", 5,
                     "Print statistics every n seconds.")
# 每隔n秒，保存训练摘要到文件。
flags.DEFINE_integer("summary_interval", 5,
                     "Save training summary to file every n seconds (rounded "
                     "up to statistics interval.")
# 每隔n秒，保存一个检查点。
flags.DEFINE_integer("checkpoint_interval", 600,
                     "Checkpoint the model (i.e. save the parameters) every n "
                     "seconds (rounded up to statistics interval.")

FLAGS = flags.FLAGS

class Options(object):

    def __init__(self):
        # 维度
        self.emb_dim = FLAGS.embedding_size

        self.train_data = FLAGS.train_data
        # Number of negative samples per example.
        # 每个样本的负采样数
        self.num_samples = FLAGS.num_neg_samples
        # 学习率
        self.learning_rate = FLAGS.learning_rate
        #要训练的epoch数。经过这么多时期，学习速率线性衰减到零，训练停止。
        self.epochs_to_train = FLAGS.epochs_to_train
        # 并发
        self.concurrent_steps = FLAGS.concurrent_steps

        self.batch_size = FLAGS.batch_size

        self.windows_size = FLAGS.windows_size

        self.min_count = FLAGS.min_count
        # 单词重现的二次采样阈值
        self.subsample = FLAGS.subsample

        self.statistics_interval = FLAGS.statistics_interval

        self.summary_interval = FLAGS.summary_interval

        self.checkpoint_interval = FLAGS.checkpoint_interval

        self.save_path = FLAGS.save_path

        self.eval_data = FLAGS.eval_data

class Word2Vec(object):
    def __init__(self,options,session):
        self._options = options
        self._session = session
        # 词和id的对应字典
        self._word2id = {}
        # id到词的对应词典
        self._id2word = []
        # 建图
        self.build_graph()

        self.build_eval_graph()
        self.save_vocab()
        self._read_analogies()

    def _read_analogies(self):
        # 读取类比文件
        questions = []
        questions_skipped = 0
        with open(self._options.eval_data,'rb') as analogy_f:
            for line in analogy_f:
                if line.startswith(b":"):
                    continue
                words = line.strip().lower().split(b" ")
                ids = [self._word2id.get(w.strip()) for w in words]
                if None in ids or len(ids) != 4:
                    questions_skipped += 1
                else:
                    questions.append(np.array(ids))
        print("Eval analogy file: ", self._options.eval_data)
        print("Questions: ", len(questions))
        print("Skipped: ", questions_skipped)
        self._analogy_questions = np.array(questions,dtype=np.int32)

    def forward(self,examples,labels):
        """建立前向传播图"""
        opts = self._options
        # 声明所有需要的变量
        # embeddings ：[vocab-size,emb_size]
        init_width = 0.5 / opts.emb_dim

        emb = tf.Variable(
            tf.random_uniform([opts.vocab_size,opts.emb_dim], -init_width,init_width),name = "emb")
        self._emb = emb

        # softmax_weights:[vocab_size,emb_dim]
        sm_w_t = tf.Variable(
            tf.zeros([opts.vocab_size,opts.emb_dim]),name="sm_w_t")
        # softmax bias:[emd_dim]
        sm_b = tf.Variable(
            tf.zeros([opts.vocab_size]),name="sm_b")

        # global step:scalar
        self.global_step = tf.Variable(0,name="global_step")

        # 候选采样计算nce loss的节点
        labels_matrix = tf.reshape(
            tf.cast(labels,dtype=tf.int64),[opts.batch_size,1])
        # 负采样
        sampled_ids, _,_ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=opts.num_samples,
            unique=True,
            range_max=opts.vocab_size,
            distortion=0.75,
            unigrams=opts.vocab_counts.tolist()))

        # 样本的嵌入:[batch_size,emb_dim]
        example_emb = tf.nn.embedding_lookup(emb,examples)

        # 标签的权重w：[batch_size,emb_dim]
        true_w = tf.nn.embedding_lookup(sm_w_t,labels)
        # 标签的偏差b ：[batch_size,1]
        true_b = tf.nn.embedding_lookup(sm_b,labels)

        # 采样样本的ids的权重(Weights for sampled ids):[num_sampled,emb_dim]
        sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
        # 采样样本的 bias :[num_sampled,1]
        sampled_b = tf.nn.embedding_lookup(sm_b,sampled_ids)

        # True logits:[batch_size,1]
        true_logits = tf.reduce_sum(tf.multiply(example_emb,true_w),1) + true_b

        # 采样样本预测值 sampled logits:[batch_size,num_sampled]
        sampled_b_vec = tf.reshape(sampled_b,[opts.num_samples])
        sampled_logits = tf.matmul(example_emb,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        return true_logits,sampled_logits

    def nce_loss(self,true_logits,sampled_logits):
        """为NCE loss构图"""
        opts = self._options
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            true_logits,tf.ones_like(true_logits)
        )
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            sampled_logits,tf.zeros_like(sampled_logits)
        )
        # nce loss 是真实值和噪声(采样词)的贡献和在batch中的平均值
        nce_loss_tensor = (tf.reduce_sum(true_xent)+tf.reduce_sum(sampled_xent)) / opts.batch_size
        return nce_loss_tensor

    def optimize(self,loss):
        """为优化损失函数构造图"""
        # 线性学习率衰减
        opts = self._options
        words_to_train = float(opts.word_per_epoch * opts.epoch_to_train)
        lr = opts.learning_rate * tf.maximum(
            0.0001,1.0-tf.cast(self._words,tf.float32)/words_to_train
        )
        self._lr = lr

        optimizer = tf.train.GradientDescentOptimizer(lr)
        train = optimizer.minimize(
            loss,global_step=self.global_step,
            gate_gradients=optimizer.GATE_NONE
        )
        self._train = train

    def build_eval_graph(self):
        """Build the eval graph."""
        # Eval graph
        # Each analogy task is to predict the 4th word (d) given three
        # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        # predict d=paris.
        # The eval feeds three vectors of word ids for a, b, c, each of
        # which is of size N, where N is the number of analogies we want to
        # evaluate in one batch.
        analogy_a = tf.placeholder(dtype=tf.int32)
        analogy_b = tf.placeholder(dtype=tf.int32)
        analogy_c = tf.placeholder(dtype=tf.int32)

        # 维度为[vocab_size,emb_dim]的正则化词向量
        nemb = tf.nn.l2_normalize(self._emb,1)

        a_emb = tf.gather(nemb,analogy_a)
        b_emb = tf.gather(nemb,analogy_b)
        c_emb = tf.gather(nemb,analogy_c)

        target = c_emb + (b_emb-a_emb)

        dist = tf.matmul(target,nemb,transpose_b=True)

        _,pred_idx = tf.nn.top_k(dist,4)

        nearby_word = tf.placeholder(dtype=tf.int32) # word id
        nearby_emb = tf.gather(nemb,nearby_word)
        nearby_dist = tf.matmul(nearby_emb,nemb,transpose_b=True)
        nearby_val,nearby_idx = tf.nn.top_k(
            nearby_dist,min(1000,self._options.vocab_size))

        self._analogy_a = analogy_a
        self._analogy_b = analogy_b
        self._analogy_c = analogy_c
        self._analogy_pred_idx = pred_idx
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx

    def build_graph(self):
        """构建模型的图"""
        opts = self._options
        # 训练数据
        (words,counts,words_per_epoch,self._epoch,self._words,examples,labels)\
            = word2vec.skipgram(filename=opts.train_data,
                            batch_size=opts.batch_size,
                            window_size=opts.window_size,
                            min_count=opts.min_count,
                            subsample=opts.subsample)

        (opts.vocab_words,opts.vocab_counts,opts.word_per_epoch)\
            = self._session.run([words,counts,words_per_epoch])
        opts.vocab_size = len(opts.vocab_words)
        print("Data file: ", opts.train_data)
        print("Vocab size: ", opts.vocab_size - 1, " + UNK")
        print("Words per epoch: ", opts.words_per_epoch)

        self._examples = examples
        self._labels = labels
        self._id2word = opts.vocab_words
        for i,w in enumerate(self._id2word):
            self._word2id[w] = i
        true_logits, sampled_logits = self.forward(examples,labels)
        loss = self.nce_loss(true_logits,sampled_logits)
        tf.summary.scalar("NCE loss",loss)
        self._loss = loss
        self.optimize(loss)

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()

    def save_vocab(self):
        """"数据存为词典，所以模型可以重载"""
        opts = self._options
        with open(os.path.join(opts.save_path,"vocab.txt"),"w") as f:
            for i in range(opts.vocab_size):
                f.write("%s %d\n" %
                        (tf.compat.as_text(opts.vocab_words[i]),opts.vocab_counts[i]))

    def _train_thread_body(self):
        initial_epoch, = self._session.run([self._epoch])
        while True:
            _,epoch = self._session.run([self._train,self._epoch])
            if epoch != initial_epoch:
                break

    def train(self):
        opts = self._options

        initial_epoch, initial_words = self._session.run([self._epoch,self._words])

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(opts.save_path,graph_def=self._session.graph_def)

        workers = []
        for _ in range(opts.concurrent_steps):
            t = threading.Thread(target=self._train_thread_body)
            t.start()
            workers.append(t)

        last_words, last_time,last_summary_time = initial_words,time.time(),0
        last_checkpoint_time = 0
        while True:
            time.sleep(opts.statistics_interval)
            (epoch,step,loss,words,lr) = self._session.run(
                [self._epoch,self.global_step,self._loss,self._words,self._lr])
            now = time.time()
            last_words,last_time,rate = words, now, (words-last_words)/(now-last_time)
            print("Epoch %4d Step %8d: lr = %5.3f loss = %6.2f words/sec = %8.0f\r" %
                (epoch, step, lr, loss, rate), end="")
            sys.stdout.flush()
            if now - last_summary_time > opts.summary_interval:
                summary_str = self._session.run(summary_op)
                summary_writer.add_summary(summary_str,step)
                last_summary_time = now
            if now -last_checkpoint_time > opts.checkpoint_interval:
                self.saver.save(self._session,opts.save_path+"model",global_step=step.astype(int))
                last_checkpoint_time = now
            if epoch != initial_epoch:
                break
        for t in workers:
            t.join()
        return epoch

    def _predict(self,analogy):
        idx, = self._session.run(
            [self._analogy_pred_idx],{
                self._analogy_a:analogy[:,0],
                self._analogy_b:analogy[:,1],
                self._analogy_c:analogy[:,2]
            })
        return idx

    def eval(self):
        correct = 0

        total = self._analogy_questions.shape[0]
        start = 0
        while start <total:
            limit = start + 2500
            sub = self._analogy_questions[start:limit, :]
            idx = self._predict(sub)
            start = limit
            for question in range(sub.shape[0]):
                for j in range(4):
                    if idx[question,j] == sub[question,3]:
                        correct += 1
                        break
                    elif idx[question,j] in sub[question,:3]:
                        continue
                    else:
                        break
        print()
        print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                                  correct * 100.0 / total))

    def analogy(self,w0,w1,w2):
        """Predict word w3 as in w0:w1 vs w2:w3."""
        wid = np.array([[self._word2id.get(w,0) for w in [w0,w1,w2]]])
        idx = self._predict(wid)
        for c in [self._id2word[i] for i in idx[0,:]]:
            if c not in [w0,w1,w2]:
                return c
        return "unknow"

    def nearby(self,words,num=20):
        """Prints out nearby words given a list of words."""
        ids = np.array([self._word2id.get(x,0) for x in words])
        vals, idx = self._session.run(
            [self._nearby_val,self._nearby_idx],{self._nearby_word:ids})
        for i in range(len(words)):
            print("\n%s\n=====================================" % (words[i]))
            for (neighbor,distance) in zip(idx[i,:num],vals[i,:num]):
                print("%-20s %6.4f" % (self._id2word[neighbor],distance))

def _start_shell(local_ns=None):
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[],user_ns=user_ns)

def main(_):
    if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
        print("--train_data --eval_data and --save_path must be specified.")
        sys.exit(1)
    opts = Options()
    with tf.Graph().as_default(),tf.Session() as session:
        model = Word2Vec(opts,session)
        for _ in range(opts.epochs_to_train):
            model.train()
            model.eval()
        model.saver.save(session,
                         os.path.join(opts.save_path,"model.ckpt"),
                         global_step=model.global_step)
        if FLAGS.interactive:
            _start_shell(locals())

if __name__ == '__main__':
    tf.app.run()

