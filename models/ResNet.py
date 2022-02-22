#!usr/bin/env python
# coding utf-8
"""
@File       :ResNet.py
@Copyright  :CV Group
@Date       :2/22/2022
@Author     :Rui
@Desc       :
"""

import tensorflow as tf


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, input_channels, output_channels, stride, expand_factor=1):
        """
        implement bottleneck blocks
        :param input_channels: number of input_channels
        :param output_channels: number of output channels
        :param stride: stride to control spatial property
        :param expand_factor: inner feature expansion factor, default set to 1
        """
        super(BottleNeck, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.conv1 = tf.keras.layers.Conv2D(filters=self.input_channels,
                                            kernel_size=(1, 1),
                                            padding='same',
                                            strides=1)
        self.conv2 = tf.keras.layers.Conv2D(filters=self.input_channels,
                                            kernel_size=(3, 3),
                                            padding='same',
                                            strides=stride)
        self.conv3 = tf.keras.layers.Conv2D(filters=self.output_channels,
                                            kernel_size=(1, 1),
                                            padding='same',
                                            strides=1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=self.output_channels,
                                                   kernel_size=(1, 1),
                                                   padding='same',
                                                   strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        res = inputs
        if self.stride != 1 and self.input_channels != self.output_channels:
            res = self.downsample(inputs)
        x += res
        x = tf.nn.relu(x)

        return x


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, channels, stride=1):
        super(BasicBlock, self).__init__()
        self.channels = channels
        self.conv1 = tf.keras.layers.Conv2D(filters=self.channels,
                                            kernel_size=(3, 3),
                                            padding='same',
                                            strides=1)

        self.conv2 = tf.keras.layers.Conv2D(filters=self.channels,
                                            kernel_size=(3, 3),
                                            padding='same',
                                            strides=stride)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=self.channels,
                                                   kernel_size=(1, 1),
                                                   padding='same',
                                                   strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.conv2(inputs)
        x = self.bn2(x)

        res = inputs
        if self.stride != 1 and self.input_channels != self.output_channels:
            res = self.downsample(inputs)
        x += res
        x = tf.nn.relu(x)

        return x

class ResNet(tf.keras.Model):
    def __init__(self, num_layers=18):
        super(ResNet, self).__init__()

