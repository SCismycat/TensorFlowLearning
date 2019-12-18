#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.26 11:19

from __future__ import print_function
import tensorflow as tf

# 创建一个常量op，这个操作符会作为一个节点加入到默认图
hello = tf.constant('Hello,TensorFlow')

sess = tf.Session()
print(sess.run(hello))
