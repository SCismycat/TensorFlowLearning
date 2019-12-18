#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.29 10:46

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

encoding_dim = 32

input_img = Input(shape=(784,))
encoded = Dense(encoding_dim,activation='relu')(input_img)
# 稀疏自编码器： 对隐层单元加稀疏性约束
# encoded = Dense(encoding_dim,activation='relu',
#                 activity_regularizer=regularizers.l2(10e-5))(input_img)

decoded = Dense(784,activation='sigmoid')(encoded)

autoencoder = Model(input_img,output=decoded)
## 单独使用编码器和解码器
encoder = Model(input_img,encoded)
encoded_input = Input(shape=(encoding_dim,))
decoded_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input,output=decoded_layer(encoded_input))

autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')

from keras.datasets import mnist

import numpy as np

(x_train,_),(x_test,_) = mnist.load_data()

x_train = x_train.astype('float32') /255.
x_test = x_test.astype('float32') /255.

x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train,x_train,nb_epoch=50,batch_size=128,shuffle=True,
                validation_data=(x_test,x_test))

import matplotlib.pyplot as plt

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(5,5,i)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # diaplay  预测的
    ax = plt.subplot(5,5, i+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()




