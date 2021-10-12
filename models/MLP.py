#!usr/bin/env python
# coding utf-8
'''
@File       :MLP.py
@Copyright  :CV Group
@Date       :9/22/2021
@Author     :Rui
@Desc       :
'''
import tensorflow as tf

class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        # Flatten 层将除第一维 batchsize 以外的维度展平
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=6)

    def call(self, inputs, training=None, mask=None):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = tf.nn.softmax(x)
        return output

