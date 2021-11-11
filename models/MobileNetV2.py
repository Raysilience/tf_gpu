#!usr/bin/env python
# coding utf-8
'''
@File       :MobileNetV2.py
@Copyright  :CV Group
@Date       :11/9/2021
@Author     :Rui
@Desc       :
'''
import tensorflow as tf

from scripts.config import NUM_CLASSES


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, input_channels, output_channels, expand_factor, stride):
        super(Bottleneck, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.expand_factor = expand_factor
        self.stride = stride
        self.conv1 = tf.keras.layers.Conv2D(
            filters=input_channels * expand_factor,
            kernel_size=(1, 1),
            strides=1,
            padding='same'
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv_dw = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=stride,
            padding='same'
        )
        self.bn_dw = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=output_channels,
            kernel_size=(1, 1),
            strides=1,
            padding='same'
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.linear = tf.keras.layers.Activation(tf.keras.activations.linear)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu6(x)

        x = self.conv_dw(x)
        x = self.bn_dw(x, training=training)
        x = tf.nn.relu6(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.linear(x)

        # only when the channels and size don't change, add a skip connection
        if self.stride == 1 and self.input_channels == self.output_channels:
            x = tf.keras.layers.add([x, inputs])
        return x

class MobileNetV2(tf.keras.Model):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=2,
            padding='same'
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bottleneck1 = self._build_bottleneck(
            input_channels=32,
            output_channels=16,
            expand_factor=1,
            stride=1,
            repeat_num=1
        )
        self.bottleneck2 = self._build_bottleneck(
            input_channels=16,
            output_channels=24,
            expand_factor=6,
            stride=2,
            repeat_num=2
        )
        self.bottleneck3 = self._build_bottleneck(
            input_channels=24,
            output_channels=32,
            expand_factor=6,
            stride=2,
            repeat_num=3
        )
        self.bottleneck4 = self._build_bottleneck(
            input_channels=32,
            output_channels=64,
            expand_factor=6,
            stride=2,
            repeat_num=4
        )
        self.bottleneck5 = self._build_bottleneck(
            input_channels=64,
            output_channels=96,
            expand_factor=6,
            stride=1,
            repeat_num=3
        )
        self.bottleneck6 = self._build_bottleneck(
            input_channels=96,
            output_channels=160,
            expand_factor=6,
            stride=2,
            repeat_num=3
        )
        self.bottleneck7 = self._build_bottleneck(
            input_channels=160,
            output_channels=320,
            expand_factor=6,
            stride=1,
            repeat_num=1
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=1280,
            kernel_size=(1, 1),
            strides=1,
            padding='same'
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))
        self.global_avg_pool = tf.keras.layers.GlobalAvgPool2D()

        self.conv3 = tf.keras.layers.Conv2D(
            filters=NUM_CLASSES,
            kernel_size=(1, 1),
            strides=1,
            padding='same'
        )

        self.fc = tf.keras.layers.Dense(
            units=NUM_CLASSES,
            activation=tf.keras.activations.softmax
        )
        self.fc1 = tf.keras.layers.Dense(
            units=12
        )

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu6(x)

        x = self.bottleneck1(x, training=training)
        x = self.bottleneck2(x, training=training)
        x = self.bottleneck3(x, training=training)
        x = self.bottleneck4(x, training=training)
        x = self.bottleneck5(x, training=training)
        x = self.bottleneck6(x, training=training)
        x = self.bottleneck7(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu6(x)

        # x = self.avg_pool(x)
        # x = self.conv3(x)
        # x = tf.nn.softmax(x)
        # return x

        x = self.global_avg_pool(x)
        y = self.fc1(x)
        x = self.fc(x)

        return x, y

    def _build_bottleneck(self, input_channels, output_channels, expand_factor, stride, repeat_num):
        block = tf.keras.Sequential()
        for i in range(repeat_num):
            if i == 0:
                block.add(Bottleneck(input_channels, output_channels, expand_factor, stride))
            else:
                block.add(Bottleneck(output_channels, output_channels, expand_factor, 1))
        return block


