#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.5 11:30
import tensorflower as tf
import numpy as np




def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              scope="embedding",
              reuse=None):
    with tf.variable_scope(scope,reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size,num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1,num_units]),lookup_table[1:,:]),0)
        outputs = tf.nn.embedding_lookup(lookup_table,inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    return outputs

def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    N,T = inputs.get_shape().as_list()
    with tf.variable_scope(scope,reuse=True):
        postion_ind = tf.tile(tf.expand_dims(tf.range(T),0),[N,1])
        postion_enc = np.array([[pos/np.power(10000,2.*i/num_units) for i in range(num_units)] for pos in range(T)])
        postion_enc[:,0::2] = np.sin(postion_enc[:,0::2])
        postion_enc[:,1::2] = np.cos(postion_enc[:,1::2])

        lookup_table = tf.convert_to_tensor(postion_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1,num_units]),lookup_table[1:,:]),0)
        outputs = tf.nn.embedding_lookup(lookup_table,postion_ind)

        if scale:
            outputs = outputs * num_units ** 0.5
    return outputs

#
def scaled_dotproduct_attention(queries,keys,num_unit=None,
                                num_heads=0,
                                dropout_rate=0,
                                is_tranining=True,causality=False,
                                scope="scaled_att",reuse=None):
    with tf.variable_scope(scope,reuse=reuse):
        if num_unit is None:
            num_unit = queries.get_shape().as_list[-1]
        # 线性变换
        Q = tf.layers.dense(queries,num_unit,activation=tf.nn.relu)
        K = tf.layers.dense(keys,num_unit,activation=tf.nn.relu)
        V = tf.layers.dense(keys,num_unit,activation=tf.nn.relu)

        outputs = tf.matmul(Q,tf.transpose(K,[0,2,1]))
        outputs = outputs/ (K.get_shape().as_list()[-1] ** 0.5)

        # 对填充的部分进行mask，这些位置att score变得极小，
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys,axis=-1)))
        key_masks = tf.tile(tf.expand_dims(key_masks,1),[1,tf.shape(queries)[1],1])

        paddings = tf.ones_like(outputs) * (-2**32 + 1)
        outputs = tf.where(tf.equal(key_masks,0),paddings,outputs)

        # 一个mask操作，对模型屏蔽未来信息
        if causality:
            diag_vals = tf.ones_like(outputs[0,:,:])
            tril = tf.contrib.linalg.LinearOperatorTril(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril,0),[tf.shape(outputs)[0],1,1])

            paddings = tf.ones_like(masks) * (-2 ** 32 +1)
            outputs = tf.where(tf.equal(masks,0),paddings,outputs)
        outputs = tf.nn.softmax(outputs)
        # Query mask
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries,axis=-1)))
        query_masks = tf.tile(tf.expand_dims(query_masks,-1),[1,1,tf.shape(keys)[1]])
        outputs *= query_masks
        outputs = tf.layers.dropout(outputs,rate=dropout_rate,training=tf.convert_to_tensor(is_tranining))
        # 加权平均
        outputs = tf.matmul(outputs,V)
        #
        outputs += queries
        outputs = normalize(outputs)
    return outputs

def multihead_attention(queries,keys,num_units=None,
                        num_heads=0,dropout_rate=0,is_training=True,
                        causality=False,scope="multihead_attention",reuse=None):
    with tf.variable_scope(scope,reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        # linear projection
        Q = tf.layers.dense(queries,num_units,activation=tf.nn.relu)
        K = tf.layers.dense(keys,num_units,activation=tf.nn.relu)
        V = tf.layers.dense(keys,num_units,activation=tf.nn.relu)
        # split and concat
        Q_ = tf.concat(tf.split(Q,num_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(K,num_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(V,num_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0,2,1]))
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        # mask
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys,axis=-1)))
        key_masks = tf.tile(key_masks,[num_heads,1])
        key_masks = tf.tile(tf.expand_dims(key_masks,1),[1,tf.shape(queries)[1],1])

        paddings = tf.ones_like(outputs) * (-2 **32 +1)
        outputs = tf.where(tf.equal(key_masks,0),paddings,outputs)

        # masked from future
        if causality:
            diag_vals = tf.ones_like(outputs[0,:,:])
            tril = tf.contrib.linalg.LinearOperatorTril(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril,0),[tf.shape(outputs)[0],1,1])

            paddings = tf.ones_like(masks) * (-2**32 +1)
            outputs = tf.where(tf.equal(masks,0),paddings,outputs)
        outputs = tf.nn.softmax(outputs)

        # query mask
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries,axis=-1)))
        query_masks = tf.tile(query_masks,[num_heads,1])
        query_masks = tf.tile(tf.expand_dims(query_masks,-1),[1,1,tf.shape(keys)[1]])
        outputs *= query_masks

        outputs = tf.layers.dropout(outputs,rate=dropout_rate,training=tf.convert_to_tensor(is_training))

        outputs = tf.matmul(outputs,V_)
        # restore shape
        outputs = tf.concat(tf.split(outputs,num_heads,axis=0),axis=2)

        outputs += queries
        outputs = normalize(outputs)
    return outputs

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    with tf.variable_scope(scope,reuse=reuse):
        inputs_shape = inputs.get_shape()
        param_shape = inputs_shape[-1:]

        mean,variance = tf.nn.moments(inputs,[-1],keep_dims=True)
        beta = tf.Variable(tf.zeros(param_shape))
        gamma = tf.Variable(tf.ones(param_shape))
        normalized = (inputs-mean)/((variance+epsilon) ** (.5))
        outputs = gamma *normalized+beta

    return outputs

def feedforward(inputs,
                num_units=[2048,512],
                scope="forward",
                reuse=None):
    with tf.variable_scope(scope,reuse=reuse):
        params = {"inputs":inputs,"filters":num_units[0],"kernel_size":1,
                  "activation":tf.nn.relu,"use_bias":True}
        outputs = tf.layers.conv1d(**params)

        # readout layer
        params = {"inputs": inputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        outputs += inputs
        outputs = normalize(outputs)
    return outputs



def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.
    For example,
    ```
    import tensorflower as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
    outputs = label_smoothing(inputs)
    with tf.Session() as sess:
        print(sess.run([outputs]))
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)










