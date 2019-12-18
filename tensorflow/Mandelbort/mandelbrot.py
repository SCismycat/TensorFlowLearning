#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.11.25 15:51

import tensorflow as tf
import numpy as np
import PIL.Image
import PIL.ImageDraw

import matplotlib.pyplot as plt
import PIL.Image
from io import StringIO
from IPython.display import clear_output,Image,display
import scipy.ndimage as nd

def DisplayFractal(a,fmt='jpeg'):
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
                          30+50*np.sin(a_cyclic),155-80*np.cos(a_cyclic)],2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a,0,255))
    img1 = PIL.Image.fromarray(a)
    plt.imsave("image_tf.png", img1)
    plt.show()

sess = tf.InteractiveSession()

Y,X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Z = X + 1j*Y
xs = tf.constant(Z.astype("complex64"))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs,"float32"))

tf.initialize_all_variables().run()

zs_ = zs*zs +xs

not_disverged = tf.abs(zs_) < 4
step = tf.group(zs.assign(zs_),
                ns.assign_add(tf.cast(not_disverged,"float32")))
for i in range(200):
    step.run()
DisplayFractal(ns.eval())