#!usr/bin/env python
# coding utf-8
'''
@File       :CNN.py
@Copyright  :CV Group
@Date       :9/22/2021
@Author     :Rui
@Desc       :
'''
import tensorflow as tf

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # number of kernels
            kernel_size=[5, 5],     # reception field of current kernel
            padding='same',         # 'valid' or 'same'
            activation=tf.nn.relu   # activation function
        )
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],
            strides=2
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],
            strides=2
        )


        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()


        self.flatten = tf.keras.layers.Reshape(
            target_shape=(7*7*64, )
        )
        self.dense1 = tf.keras.layers.Dense(
            units=1024,
            activation=tf.nn.relu
        )
        self.dense2 = tf.keras.layers.Dense(
            units=6
        )

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
        x = self.pool1(x)                       # [batch_size, 14, 14, 32]
        x = self.conv2(x)                       # [batch_size, 14, 14, 64]
        x = self.pool2(x)                       # [batch_size, 7, 7, 64]
        x = self.conv3(x)
        x = self.avg_pool(x)
        # x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
        # x = self.dense1(x)                      # [batch_size, 1024]
        x = self.dense2(x)                      # [batch_size, 10]
        x = tf.nn.softmax(x)
        return x