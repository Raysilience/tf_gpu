#!usr/bin/env python
# coding utf-8
'''
@File       :MobileNetV1.py
@Copyright  :CV Group
@Date       :11/11/2021
@Author     :Rui
@Desc       :
'''
import tensorflow as tf

from scripts.config import NUM_CLASSES


class DepthSeparableBlock(tf.keras.layers.Layer):
    def __init__(self, output_channels, strides):
        super(DepthSeparableBlock, self).__init__()
        self.conv_dw = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=strides,
            padding='same'
        )
        self.bn_dw = tf.keras.layers.BatchNormalization()

        self.conv_pw = tf.keras.layers.Conv2D(
            filters=output_channels,
            kernel_size=(1, 1),
            strides=1,
            padding='same'
        )
        self.bn_pw = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv_dw(inputs)
        x = self.bn_dw(x)
        x = tf.nn.relu(x)
        x = self.conv_pw(x)
        x = self.bn_pw(x)
        x = tf.nn.relu(x)
        return x


class MobileNetV1(tf.keras.Model):
    def __init__(self, alpha=1):
        super(MobileNetV1, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=2,
            padding='same'
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.block1 = DepthSeparableBlock(int(64 * alpha), 1)
        self.block2 = DepthSeparableBlock(int(128 * alpha), 2)
        self.block3 = DepthSeparableBlock(int(128 * alpha), 1)
        self.block4 = DepthSeparableBlock(int(256 * alpha), 2)
        self.block5 = DepthSeparableBlock(int(256 * alpha), 1)
        self.block6 = DepthSeparableBlock(int(512 * alpha), 2)
        self.block7 = DepthSeparableBlock(int(512 * alpha), 1)
        self.block8 = DepthSeparableBlock(int(1024 * alpha), 2)
        self.block9 = DepthSeparableBlock(int(1024 * alpha), 1)
        self.avg_pool = tf.keras.layers.GlobalAvgPool2D()
        self.fc = tf.keras.layers.Dense(
            filter=NUM_CLASSES
        )


        # depth separable conv can also be implemented in the following way
        # self.conv_sp = tf.keras.layers.SeparableConv2D()

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        for i in range(5):
            x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        x = tf.nn.softmax(x)
        return x
