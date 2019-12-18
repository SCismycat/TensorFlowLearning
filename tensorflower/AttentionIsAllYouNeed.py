#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.5 14:23

import tensorflower as tf
from tqdm import tqdm
import os,codecs
from .Transformer_Demo import *
class AttentionGraph():
    def __init__(self,is_training=True):
        self.graph = tf.Graph()

        with self.graph.as_default():
            if is_training:
                self.x,self.y,self.num_batch = get_batch_data()
            else:
                self.x = tf.placeholder(tf.int32,shape=(None,hp.maxlen))
                self.y = tf.placeholder(tf.int32,shape=(None,hp.maxlen))

            # define decode inputs
            self.decode_inputs = tf.concat((tf.ones_like(self.y[:,:1])*2,self.y[:,:-1]),-1)

            de2idx, idx2de = load_de_vocab()
            en2idx, idx2de = load_en_vocab()
            with tf.variable_scope("encoder"):
                # embedding
                self.enc = embedding(self.x,
                                     vocab_size=len(de2idx),
                                     zero_pad=True,
                                     scale=True,
                                     scope="enc_embed")
                # pos embedding
                if hp.sinusoid:
                    self.enc += positional_encoding(self.x,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=True,
                                                    scale=False,
                                                    scope="enc_pos")
                else:
                    self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]),0),[tf.shape(self.x)[0],1]),
                                          vocab_size=hp.maxlen,
                                          num_units=hp.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="enc_pos")
