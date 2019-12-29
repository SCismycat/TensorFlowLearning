#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/24 20:02
# @Author  : Leslee

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("data_dir","../data/mnist_data","mnist data")
FLAGS = flags.FLAGS

def main(_):
    print(FLAGS.data_dir)


if __name__ == '__main__':
    tf.app.run()


