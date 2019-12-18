#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.26 17:10

import tensorflower as tf
from tensorflower.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
from tensorflower.contrib.boosted_trees.proto import learner_pb2 as gbdt_learner
from tensorflower.examples.tutorials.mnist import input_data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False,
                                  source_url='http://yann.lecun.com/exdb/mnist/')

batch_size = 4096
num_classes = 10
num_features = 784
max_steps = 10000

# gbdt 参数
learning_rate = 0.1
l1_regul = 0.
l2_regul = 1.
examples_per_layer = 1000
num_trees = 10
max_depth = 16

# 把GBDT参数填充到配置协议中
learner_config = gbdt_learner.LearnerConfig()
learner_config.learning_rate_tuner.fixed.learning_rate = learning_rate
learner_config.regularization.l1 = l1_regul
learner_config.regularization.l2 = l2_regul / examples_per_layer
learner_config.constraints.max_tree_depth = max_depth
growing_mode = gbdt_learner.LearnerConfig.LAYER_BY_LAYER
learner_config.growing_mode = growing_mode
run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=300)
learner_config.multi_class_strategy = (
    gbdt_learner.LearnerConfig.DIAGONAL_HESSIAN)\

# 创建tensorflow的gbdt 模型
gbdt_model = GradientBoostedDecisionTreeClassifier(
    model_dir=None,
    learner_config = learner_config,
    n_classes=num_classes,
    examples_per_layer = examples_per_layer,
    num_trees = num_trees,
    center_bias=False,
    config=run_config
    )
tf.logging.set_verbosity(tf.logging.INFO)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'image':mnist.train.images},y=mnist.train.labels,
    batch_size=batch_size,num_epochs=None,shuffle=True)

# 训练模型
gbdt_model.fit(input_fn=input_fn,max_steps=max_steps)

# 评估模型
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images':mnist.test.images},y=mnist.test.labels,
    batch_size=batch_size,shuffle=False
)
e = gbdt_model.evaluate(input_fn=input_fn)
print("Testing ACC",e['accuracy'])



















