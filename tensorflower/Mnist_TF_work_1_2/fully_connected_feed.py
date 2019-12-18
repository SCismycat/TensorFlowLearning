#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Leslee


from __future__ import absolute_import,division,print_function

import argparse
import os
import sys
import time

from six.moves import xrange
import tensorflower as tf

from com.hiekn.tensorflow.Mnist_TF_work_1_2 import input_data
from com.hiekn.tensorflow.Mnist_TF_work_1_2 import mnist

Flags = None
# 定义一个占位function
def placeholder_inputs(batch_size):
    images_placeholder = tf.compat.v1.placeholder(
        tf.float32,shape=(batch_size,mnist.IMAGE_PIXELS)
    )
    labels_placeholder = tf.compat.v1.placeholder(tf.int32,shape=(batch_size))
    return images_placeholder,labels_placeholder
# 定义一个训练数据填充的方法
def fill_feed_dict(data_set,image_pl,label_pl):
    image_feed,label_feed = data_set.next_batch(Flags.batch_size,Flags.fake_data)
    feed_dict = {
        image_pl : image_feed,
        label_pl : label_feed,
    }
    return feed_dict

def do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_set):
    true_count = 0
    step_per_epoch = data_set.num_examples // Flags.batch_size
    num_examples = step_per_epoch * Flags.batch_size
    for step in range(step_per_epoch):
        feed_dict = fill_feed_dict(data_set,images_placeholder,labels_placeholder)
        true_count += sess.run(eval_correct,feed_dict=feed_dict)
    precision = float(true_count)/num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))

def run_training():
    data_set = input_data.read_data_sets(Flags.input_data_dir,Flags.fake_data)
    with tf.Graph().as_default():
        images_placeholder,labels_placeholder = placeholder_inputs(
            Flags.batch_size
        )
        logits = mnist.inference(images_placeholder,Flags.hidden1,Flags.hidden2)
        loss = mnist.loss(logits,labels_placeholder)
        train_op = mnist.training(loss,Flags.learning_rate)

        eval_correct = mnist.evaluation(logits,labels_placeholder)

        summary = tf.compat.v1.summary.merge_all()

        init = tf.compat.v1.global_variables_initializer()

        saver = tf.compat.v1.train.Saver()

        sess = tf.compat.v1.Session()
        summary_writer = tf.compat.v1.summary.FileWriter(Flags.logdir,sess.graph)

        sess.run(init)

        for step in range(Flags.max_steps):
            start_time = time.time()

            feed_dict = fill_feed_dict(data_set.train,images_placeholder,labels_placeholder)

            _,loss_value = sess.run([train_op,loss],feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 100 ==0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary,feed_dict=feed_dict)
                summary_writer.add_summary(summary_str,step)
                summary_writer.flush()

            if step % 1000 ==0 or (step+1) ==Flags.max_steps:
                checkpoint_file = os.path.join(Flags.logdir,'model.ckpt')
                saver.save(sess,checkpoint_file,global_step=step)
                print("Training Date Eval")
                do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_set.train)
                print("valid Date Eval")
                do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_set.validation)
                print("test Date Eval")
                do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_set.test)


def main(_):
    if tf.io.gfile.exists(Flags.log_dir):
        tf.io.gfile.rmtree(Flags.log_dir)
    tf.io.gfile.makedirs(Flags.log_dir)
    run_training()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'lr',
        type = 'float',
        default= 0.01,
        help='init lr'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflower/mnist/input_data'),
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflower/mnist/logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )
    Flags,unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main,argv=[sys.argv[0]]+unparsed)


















