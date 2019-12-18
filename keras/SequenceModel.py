#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/6/28 23:19
# @Author  : Leslee
import keras
import numpy as np
from keras import Sequential
from keras.backend import dropout
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD

x_train = np.random.random((1000,20))
y_train = keras.utils.to_categorical(np.random.randint(10,size=(1000,1)), num_classes=10)
x_test = np.random.random((100,20))
y_test = keras.utils.to_categorical(np.random.randint(10,size=(100,1)),num_classes=10)

def mlp_multiclass():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=20))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # 设置sgd
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20, batch_size=128)
    print("model train ending..")
    score = model.evaluate(x_test, y_test, batch_size=128)
    print(score)


## 实现一个图像分类的卷积神经网络

def vgg_cnn(x_train1,y_train1,x_test1,y_test1):
    model1 = Sequential()
    model1.add(Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)))
    model1.add(Conv2D(32,(3,3),activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2,2)))
    model1.add(Dropout(0.25))

    model1.add(Conv2D(64,(3,3),activation='relu'))
    model1.add(Conv2D(64,(3,3),activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2,2)))
    model1.add(Dropout(0.25))

    model1.add(Flatten())
    model1.add(Dense(256,activation='relu'))
    model1.add(Dropout(0.5))
    model1.add(Dense(10,activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model1.compile(loss='categorical_crossentropy', optimizer=sgd)

    model1.fit(x_train1, y_train1, batch_size=32, epochs=10)
    score = model1.evaluate(x_test1, y_test1, batch_size=32)
    print(score)

if __name__ == '__main__':
    x_train1 = np.random.random((100, 100, 100, 3))
    y_train1 = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    x_test1 = np.random.random((20, 100, 100, 3))
    y_test1 = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)
    vgg_cnn(x_train1,y_train1,x_test1,y_test1)