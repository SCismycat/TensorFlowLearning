#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.25 11:38

from __future__ import absolute_import,division
import time
import numpy as np
import tensorflow as tf

from com.hiekn.tensorflow.RNN_ptb_tutorial import reader,rnn,rnn_cell,seq2seq

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model","small","模型大小")
flags.DEFINE_string("data_path","D:/MyDownload/simple-examples/simple-examples/data/","data_path")

FLAGS = flags.FLAGS

class PTBModel(object):

    def __init__(self,is_training,config):
        self.batch_size = batch_size = config.batch_size # batch_size
        self.num_steps = num_steps = config.num_steps #
        size = config.hidden_size # 隐藏层
        vocab_size = config.vocab_size # 词表size
        # 输入占位符
        self._input_data = tf.placeholder(tf.int32,[batch_size,num_steps])
        self._targets = tf.placeholder(tf.int32,[batch_size,num_steps])

        lstm_cell = rnn_cell.BasicLSTMCell(size,forget_bias=0.0)
        if is_training and config.keep_prob <1:
            lstm_cell = rnn_cell.DropoutWrapper(
                lstm_cell,output_keep_prob=config.keep_prob)
        cell = rnn_cell.MultiRNNCell([lstm_cell]*config.num_layers)

        self._initial_state = cell.zero_state(batch_size,tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding",[vocab_size,size])
            inputs = tf.nn.embedding_lookup(embedding,self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs,config.keep_prob)

        outputs = []
        states = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step>0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output,state) = cell(inputs[:,time_step,:],state)
                outputs.append(cell_output)
                states.append(state)

        output = tf.reshape(tf.concat(outputs,1),[-1,size])
        softmax_w = tf.get_variable("softmax_w",[size,vocab_size])
        softmax_b = tf.get_variable("softmax_b",[vocab_size])
        logits = tf.matmul(output,softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],[tf.reshape(self._targets,[-1])],
            [tf.ones([batch_size*num_steps])],vocab_size)
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = states[-1]

        if not is_training:
            return
        self._lr = tf.Variable(0.0,trainable=False)
        tvars = tf.trainable_variables()
        grads,_ = tf.clip_by_global_norm(tf.gradients(cost,tvars),
                                         config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads,tvars))

    def assign_lr(self,session,lr_value):
        session.run(tf.assign(self.lr,lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

class SmallConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

class MediumConfig(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000

class LargeConfig(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000

def run_epoch(session,m,data,eval_op,verbose=False):

    epoch_size = ((len(data) // m.batch_size)-1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    # state = m.initial_state.eval()
    # m.initial_state = tf.convert_to_tensor(m.initial_state)
    state = tf.get_default_session().run(m.initial_state)
    for step, (x,y) in enumerate(reader.ptb_iterator(data,m.batch_size,m.num_steps)):
        cost,state,_ = session.run(
            [m.cost,m.final_state,eval_op],
            {m.input_data:x,m.targets:y,m.initial_state: state})

        cost += cost
        iters += m.num_steps

        if verbose and step % (epoch_size//10) ==10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * m.batch_size / (time.time() - start_time)))

    return np.exp(costs/iters)

def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

def main(unused_args):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data,valid_data,test_data,_ = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        with tf.variable_scope("model",reuse=None,initializer=initializer):
            m = PTBModel(is_training=True,config=config)
        with tf.variable_scope("model",reuse=True,initializer=initializer):
            mvalid = PTBModel(is_training=False,config=config)
            mtest = PTBModel(is_training=False,config=eval_config)

        tf.initialize_all_variables().run()

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i-config.max_epoch,0.0)
            m.assign_lr(session,config.learning_rate*lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session,m,train_data,m.train_op,verbose=True)

            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session,mvalid,valid_data,tf.no_op())
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
        test_perplexity = run_epoch(session,mtest,test_data,tf.no_op())


if __name__ == '__main__':
    tf.compat.v1.app.run()