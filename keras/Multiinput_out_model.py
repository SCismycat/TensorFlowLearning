#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/7/1 23:17
# @Author  : Leslee
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional
from keras.models import Model
# 演示一个多输入多输出的Model
import keras
headline_data = None
additional_data = None
labels = None
# 输入层
main_input = Input(shape=(100,),dtype='int32',name='main_input')
# Embedding层
X = Embedding(output_dim=128,input_dim=10000,input_length=100)(main_input)
# LSTM对一层新闻标题进行建模
lstm_out1 = Bidirectional(LSTM(32,return_sequences=True))(X)
# 新闻标题二分类。
aux_output = Dense(1,activation='sigmoid',name='aux_output')(lstm_out1)

aux_input = Input(shape=(5,),name='aux_input')
x = keras.layers.concatenate([lstm_out1,aux_input])

x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)

main_output = Dense(1,activation='sigmoid',name='main_output')(x)
model = Model(inputs=[main_input,aux_input],outputs=[main_output,aux_output])

# 编译模型，并给辅助损失分配一个 0.2 的权重。如果要为不同的输出指定不同的 loss_weights 或 loss，可以使用列表或字典
model.compile(optimizer='rmsprop',loss='binary_crossentropy',loss_weights=[1.,0.2])

model.fit([headline_data,additional_data],[labels,labels],epochs=10,batch_size=32)
# 如果通过定义层的时候传入了name参数。也可以用以下方式编译。
model.compile(optimizer='rmsprop',loss={'main_output':'binary_crossentropy','aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)
## 使用共享网络层模型
# 场景如下：建立一个模型来判断两条推文是否来自同一个人。即建立一个模型，将两个推文编码成
# 两个向量，连接向量，添加逻辑回归层判断两个文章是否同一作者。因为这时候问题是对称的，第一条推文应该完全重用来编码第二条推文。
# 所以，使用一个共享的lstm层
tw_a = Input(shape=(280,256))
tw_b = Input(shape=(280,256))

shared_lstm = LSTM(64)

encode_a = shared_lstm(tw_a)
encode_b = shared_lstm(tw_b)

merge_vec = keras.layers.concatenate([encode_a,encode_b],axis=-1)

# 增加逻辑回归层判断是否是同一条。
predict = Dense(1,activation='sigmoid')(merge_vec)

model = Model(inputs=[tw_a,tw_b],outputs=predict)
model.compile(optimizer='rmsprop',loss= 'binary_crossentropy',
              metrics=['accuracy'])
model.fit([],labels,epochs=10)
# 层[节点]的概念：当多个输入调用一个层的时候，图层会形成多个节点的索引，0，1,2...可以通过索引来访问
