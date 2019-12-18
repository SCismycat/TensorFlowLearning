#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 0:40
# @Author  : Leslee
# 残差链接
from keras.layers import Conv2D, Input, MaxPooling2D, Flatten

x = Input(shape=(256,256,3))
y = Conv2D(3,(3,3),padding='same')(x)
import keras
z = keras.layers.add([x,y])

# 视觉问答model
main_input = Input(None,None,)

model = Conv2D(64,(3,3),activation='relu',padding='same')(main_input)

model = Conv2D(64,(3,3),activation='relu')(model)

model = MaxPooling2D((2,2))(model)

model = Conv2D(256,(3,3),activation='relu',padding='same')(model)

model = Conv2D(256,(3,3),activation='relu')(model)
model = MaxPooling2D((2,2))(model)
model = Flatten(model)
input_img = None
encode_img = model(input_img)

ques_input = None
input_ques  = None
encode_ques = model(input_ques)
