#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/25 11:44
# @Author  : Leslee

from __future__ import absolute_import,division,print_function

import math
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow import flags
flags = tf.flags
flags.DEFINE_string("data_dir","../data/MNIST_data","mnist data")
flags.DEFINE_string("train_dir","../model/mnist_log","storing checkpoint and summary")
# 任务索引，如果任务是0.表示对变量进行初始化。
flags.DEFINE_integer("task_index",0,"work task index ,应该要求大于等于0"
                                      "等于0表示对变量执行初始化")
# GPU数量，0表示没有。
flags.DEFINE_integer("num_gpus",1,"total num of gpus")
#
flags.DEFINE_integer("replicas_to_aggregate",None,
                     "允许参数更新要聚合的副本数，即训练数据数量分成多少份。仅对同步训练班模式有效"
                     "默认是num_workers")
flags.DEFINE_integer("hidden_units",100,"隐藏层个数")
flags.DEFINE_integer("train_steps",200,"训练次数")
flags.DEFINE_integer("batch_size",100,"训练的batch_size")
flags.DEFINE_float("learning_rate",0.01,"learn rate")
flags.DEFINE_boolean("sync_replicas",False,"是否使用同步训练机制")
flags.DEFINE_string("ps_hosts","localhost:2223","PS服务器")
flags.DEFINE_string("worker_hosts","localhost:2224","workers服务器")
flags.DEFINE_string("job_name","worker","job name:worker or ps")

FLAGS = flags.FLAGS

IMAGE_PIXELS = 28

def main(unused_argv):
    mnist = input_data.read_data_sets(FLAGS.data_dir,one_hot=True)

    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("must specify job_name")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an task_index")

    # 解析ps和worker的主机名列表
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")

    # 计算worker的数量
    num_workers = len(worker_spec)

    cluster = tf.train.ClusterSpec({
        "ps":ps_spec,
        "worker":worker_spec
    })

    server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    # 如果是PS，直接启动服务，开始监听workers发起的请求
    if FLAGS.job_name == "ps":
        server.join()
    # 判断当前是否是chief worker(编号为0的worker，负责初始化、恢复和保存全局变量)
    is_chief = (FLAGS.task_index == 0)

    if FLAGS.num_gpus > 0:
        # 假设每台机器的GPU相同，则为每台GPU的每个GPU分配一个计算任务。
        gpu = (FLAGS.task_index % FLAGS.num_gpus)
        worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index,gpu)
    elif FLAGS.num_gpus == 0:
        # 如果没有 GPU，直接将计算任务分配到 CPU
        cpu = 0
        worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)

    # 根据TensorFlow的集群定义和当前设备信息，放置对应的模型参数和计算操作
    with tf.device(
        # 设备设置器，接受以下三个参数(任务绑定的设备名称，PS任务绑定名称，和集群)
        tf.train.replica_device_setter(
            worker_device=worker_device,
            ps_device= "/job:ps/cpu:0",
            cluster=cluster)):
        # 同步需要使用的global——step
        global_step = tf.Variable(0,name="global_step",trainable=False)

        hid_w = tf.Variable(
            tf.truncated_normal(
                [IMAGE_PIXELS*IMAGE_PIXELS,FLAGS.hidden_units],stddev=1.0/IMAGE_PIXELS),
            name="hid_w")
        hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]),name='hid_b')

        # softmax
        sm_w = tf.Variable(tf.truncated_normal(
            [FLAGS.hidden_units,10],
            stddev=1.0/math.sqrt(FLAGS.hidden_units)),
            name='hid_w')
        sm_b = tf.Variable(tf.zeros([10]),name='sm_b')

        x = tf.placeholder(tf.float32,[None,IMAGE_PIXELS*IMAGE_PIXELS])
        y_ = tf.placeholder(tf.float32,[None,10])

        hidden_lin = tf.nn.xw_plus_b(x,hid_w,hid_b)
        hid = tf.nn.relu(hidden_lin)
        y = tf.nn.softmax(tf.nn.xw_plus_b(hid,sm_w,sm_b))

        corss_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))

        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

    if FLAGS.sync_replicas:
        # 如果没有并行副本数，则为默认值num_workers
        if FLAGS.replicas_to_aggregate is None:
            replicas_to_aggregate = num_workers
        else:
            replicas_to_aggregate = FLAGS.replicas_to_aggregate
        # 创建同步优化器实例，负责梯度计算和模型参数更新
        opt = tf.train.SyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=replicas_to_aggregate,
            total_num_replicas=num_workers,
            name="mnist_sync_replicas")
    train_op = opt.minimize(corss_entropy,global_step=global_step)

    # 如果使用同步训练机制，需要保证local_step的所有机器相同的global_step
    if FLAGS.sync_replicas:
        # 其他worker：为local_step设置初始值
        local_init_op = opt.local_step_init_op
        # chief worker：为global_step设置初始值
        if is_chief:
            local_init_op = opt.chief_init_op

        # 定义为未初始化的Variable设置初始值操作
        ready_for_local_init_op = opt.ready_for_local_init_op

        # 定义启动同步标记队列的QueueRunner实例
        chief_queue_runner = opt.get_chief_queue_runner()
        # 定义为同步标记队列入队初始值的操作
        sync_init_op = opt.get_init_tokens_op()
    # 定义为全局Variable设置初始值的操作
    init_op = tf.global_variables_initializer()
    # 传入本地初始化相关操作
    if FLAGS.sync_replicas:
        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=FLAGS.train_dir,
            init_op = init_op,
            local_init_op=local_init_op,
            ready_for_local_init_op=ready_for_local_init_op,
            recovery_wait_secs=1,
            global_step=global_step)
    # 异步
    else:
        sv = tf.train.Supervisor(
          is_chief=is_chief,
          logdir=FLAGS.train_dir,
          init_op=init_op,
          recovery_wait_secs=1,
          global_step=global_step)
    # 配置分布式会话Session：
    # 没有可用GPU的话，操作放在CPU。
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps","/job:worker/task:%d" % FLAGS.task_index])

    # 如果是chief workers ,初始化所有worker的分布式会话
    if is_chief:
        print("Worker %d: Initializing session..." % FLAGS.task_index)
    else:
        print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.task_index)
    sess = sv.prepare_or_wait_for_session(server.target,config=sess_config)
    print("Worker %d: Session initialization complete." % FLAGS.task_index)
    # 如果是同步模式，并且当前进程是chief worker
    if FLAGS.sync_replicas and is_chief:
        # 初始化同步标记队列
        sess.run(sync_init_op)
        # 通过queue runner 启动三个线程，分别运行各自的服务
        sv.start_queue_runners(sess,[chief_queue_runner])

    # 记录并且打印训练开始前的时间
    time_begin = time.time()
    print("Training begins @ %f" % time_begin)
    local_step = 0
    while True:
        # 填充训练数据
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        train_feed = {x: batch_xs, y_: batch_ys}
        # 执行单步训练操作
        _,step = sess.run([train_op,global_step],feed_dict=train_feed)
        local_step += 1
        now = time.time()
        print("%f: Worker %d: training step %d done (global step: %d)" %
              (now, FLAGS.task_index, local_step, step))

        if step>=FLAGS.train_steps:
            break
    # 记录并打印训练结束的时间
    time_end = time.time()
    print("Training ends @ %f" % time_end)
    # 总训练时间为两者的时间差
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)

    val_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
    # 验证集上计算模型交叉熵
    val_xgent = sess.run(corss_entropy,feed_dict=val_feed)
    print("After %d training step(s), validation cross entropy = %g" %
          (FLAGS.train_steps, val_xgent))

if __name__ == '__main__':
    tf.app.run()