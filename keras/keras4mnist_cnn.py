#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Leslee
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import matplotlib.pyplot as plt
np.random.seed(10)
(x_train_image,y_train_label),\
(x_test_image,y_test_label) = mnist.load_data()
print('trian data length=',len(x_train_image))
print('test data length=',len(x_test_image))# length

print('x_train image size：',x_train_image.shape)
print('y_train-label：',y_train_label.shape)#size

# 转换成四维矩阵，成为cnn的输入
x_Train4D = x_train_image.reshape(x_train_image.shape[0],28,28,1).astype('float32')
x_Test4D = x_test_image.reshape(x_test_image.shape[0],28,28,1).astype('float32')
# 归一化
x_Train4D_norm = x_Train4D/255
x_Test4D_norm = x_Test4D/255

y_train_onehot = np_utils.to_categorical(y_train_label)
y_test_onehot = np_utils.to_categorical(y_test_label)

# 建立模型
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history = model.fit(x=x_Train4D_norm,y=y_train_onehot,validation_split=0.2,epochs=10,batch_size=500,verbose=2)
# 画图像
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('trian history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()
show_train_history(train_history,'loss','val_loss')
scores = model.evaluate(x_Test4D_norm,y_test_onehot)
print(scores[1])

prediction = model.predict_classes(x_Test4D_norm)
print(prediction[:10])














