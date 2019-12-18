#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.5 17:00
import keras
from keras.layers import Input, Embedding, LSTM,Dense
from keras.models import Model

main_input = Input(shape=(100,),dtype='int32',name='main_input')

X = Embedding(output_dim=512,input_dim=10000,input_length=1000)(main_input)

lstm_out = LSTM(32)(X)

auxiliary_output = Dense(1,activation='sigmoid',name='aux_output')(lstm_out)

auxiliary_input = Input(shape=(5,),name='aux_input')
X = keras.layers.concatenate([lstm_out,auxiliary_input])

X = Dense(64,activation='relu')(X)
X = Dense(64,activation='relu')(X)
X = Dense(64,activation='relu')(X)

main_out = Dense(1,activation='sigmoid',name='main_output')(X)

model = Model(inputs=[main_input,auxiliary_input], outputs=[main_out,auxiliary_output])
model.compile(optimizer='rmsprop',loss='binary_crossentropy',loss_weights=[1.,0.2])

model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)







