#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 15:52
# @Author  : Leslee
import tensorflow as tf
from tensorflow import flags
FLAGS = tf.app.flags
# 创建TF集群参数
flags.DEFINE_integer("task_index",None,"work Task Idx") # 任务名称，worker的任务编号
flags.DEFINE_string("ps_hosts",None,"ip-port") # PS的主机名
flags.DEFINE_string("work_hosts",None,"host_name") # worker的主机名
flags.DEFINE_string("job_name",None,"Job name: worker or PS") # 作业名称worker还是PS

def main(unused_argv):
    # 解析参数ps_hosts和work_hosts
    PS_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    # 定义TensorFlow集群
    cluster = tf.train.ClusterSpec({
        "PS":PS_spec,
        "worker":worker_spec})

    server = tf.train.Server(cluster,job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    # PS任务，开始监听各worker的请求
    if FLAGS.job_name == "PS":
        server.join()
    # 如果是worker任务，则把任务编号为0的worker设置为chief worker
    is_chief = (FLAGS.task_index == 0)