#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Leslee
import numpy as np
import pandas as pd
from keras.utils import np_utils
np.random.seed(10)
from keras.datasets import mnist
import matplotlib.pyplot as plt
(x_train_image,y_train_label),\
(x_test_image,y_test_label) = mnist.load_data()
print('trian data length=',len(x_train_image))
print('test data length=',len(x_test_image))# length

print('x_train image size：',x_train_image.shape)
print('y_train-label：',y_train_label.shape)#size

x_train = x_train_image.reshape(60000,784).astype('float32')
x_test = x_test_image.reshape(10000,784).astype('float32')

x_train_norm = x_train/255
x_test_norm = x_test/255

y_train_cate = np_utils.to_categorical(y_train_label)
y_test_cate = np_utils.to_categorical(y_test_label)
# print(y_train_cate[:5])

from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(units=1000,
                input_dim=784,
                kernel_initializer='normal',
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1000,
                kernel_initializer='normal',
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
train_history = model.fit(x=x_train_norm,
                          y=y_train_cate,validation_split=0.2,
                          epochs=10,
                          batch_size=200,
                          verbose=2)
scores = model.evaluate(x_test_norm,y_test_cate)
# 进行预测
prediction = model.predict_classes(x_test)
print(prediction)
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25:
        num=25
    for i in range(0,num):
        ax = plt.subplot(5,5,1+i)
        ax.imshow(images[idx],cmap='binary')
        title = "label=" + str(labels[idx])
        if len(prediction)>0:
            title +=",predict="+str(prediction[idx])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx +=1
    plt.show()


def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('trian history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

pd.crosstab(y_test_label,prediction,rownames=['labels'],colnames=['predict'])
df = pd.DataFrame({'label':y_test_label,'predict':prediction})
print(df[:50])
print(df[(df.label==5)&(df.predict==3)])


# 显示数字图像
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image,cmap='binary')
    plt.show()
    print(y_train_label[0])
# 多显示点图像
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25:
        num=25
    for i in range(0,num):
        ax = plt.subplot(5,5,1+i)
        ax.imshow(images[idx],cmap='binary')
        title = "label=" + str(labels[idx])
        if len(prediction)>0:
            title +=",predict="+str(prediction[idx])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx +=1
    plt.show()

if __name__ == '__main__':
    # plot_image(x_train_image[0])
    # plot_images_labels_prediction(x_train_image,y_train_label,[],0,10)
    show_train_history(train_history, 'acc', 'val_acc')# 准确率执行结果
    show_train_history(train_history, 'loss', 'val_acc')# 误差
    plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx=340)
    plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx=340,num=1)