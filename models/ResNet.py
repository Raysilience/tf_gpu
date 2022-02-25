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
    def __init__(self, input_channels, output_channels, stride, shortcut_dim_raising=False):
        """
        implement bottleneck blocks
        :param input_channels: number of input_channels
        :param output_channels: number of output channels
        :param stride: stride to control spatial property
        :param shortcut_dim_raising: raise the dimension of shortcut default set to False. only used for the very first layer.
        """
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=input_channels,
                                            kernel_size=(1, 1),
                                            padding='same',
                                            strides=1)
        self.conv2 = tf.keras.layers.Conv2D(filters=input_channels,
                                            kernel_size=(3, 3),
                                            padding='same',
                                            strides=stride)
        self.conv3 = tf.keras.layers.Conv2D(filters=output_channels,
                                            kernel_size=(1, 1),
                                            padding='same',
                                            strides=1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()

        if stride == 1 and not shortcut_dim_raising:
            self.down_sample = lambda x, training: x
        else:
            self.down_sample = tf.keras.Sequential()
            self.down_sample.add(tf.keras.layers.Conv2D(filters=output_channels,
                                                        kernel_size=(1, 1),
                                                        padding='same',
                                                        strides=stride))
            self.down_sample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        residual = self.down_sample(inputs, training=training)
        x += residual
        x = tf.nn.relu(x)

        return x


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, output_channels, stride, shortcut_dim_raising=False):
        """
        build basic block with two consecutive conv3*3
        :param input_channels:
        :param output_channels:
        :param stride: default set to 1
        :param shortcut_dim_raising: raise the dimension of shortcut default set to False. not used in the resnet paper.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=input_channels,
                                            kernel_size=(3, 3),
                                            padding='same',
                                            strides=stride)

        self.conv2 = tf.keras.layers.Conv2D(filters=output_channels,
                                            kernel_size=(3, 3),
                                            padding='same',
                                            strides=1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride == 1 and not shortcut_dim_raising:
            self.down_sample = lambda x, training: x
        else:
            self.down_sample = tf.keras.Sequential()
            self.down_sample.add(tf.keras.layers.Conv2D(filters=output_channels,
                                                        kernel_size=(1, 1),
                                                        padding='same',
                                                        strides=stride))
            self.down_sample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        residual = self.down_sample(inputs, training=training)
        x += residual
        x = tf.nn.relu(x)
        return x


def _build_blocks(block_type, input_channels, output_channels, stride, repeat_num, shortcut_dim_raising):
    layer = tf.keras.Sequential()
    layer.add(block_type(input_channels, output_channels, stride, shortcut_dim_raising))
    for _ in range(1, repeat_num):
        layer.add(block_type(input_channels, output_channels, 1))
    return layer


class ResNet(tf.keras.Model):
    def __init__(self, num_classes, block_type, layer_params, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        if 'name' in kwargs:
            self.model_name = kwargs.get('name')

        self.block_type = block_type
        self.layer_params = layer_params
        self.conv = tf.keras.layers.Conv2D(filters=64,
                                           kernel_size=(7, 7),
                                           padding='same',
                                           strides=2)
        self.bn = tf.keras.layers.BatchNormalization()
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                  padding='same',
                                                  strides=2)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=num_classes,
                                        activation=tf.keras.activations.softmax)

        self.shortcut_dim_raising = True
        self.blocks = []
        for param in layer_params:
            self.blocks.append(_build_blocks(self.block_type, param[0], param[1], param[2], param[3], self.shortcut_dim_raising))
            self.shortcut_dim_raising = False

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x)
        x = self.fc(x)
        return x


def resnet18(num_classes, **kwargs):
    layer1 = [64, 64, 1, 2]
    layer2 = [128, 128, 2, 2]
    layer3 = [256, 256, 2, 2]
    layer4 = [512, 512, 2, 2]
    layer_params = [layer1, layer2, layer3, layer4]
    return ResNet(num_classes, BasicBlock, layer_params, **kwargs)


def resnet34(num_classes, **kwargs):
    layer1 = [64, 64, 1, 3]
    layer2 = [128, 128, 2, 4]
    layer3 = [256, 256, 2, 6]
    layer4 = [512, 512, 2, 3]
    layer_params = [layer1, layer2, layer3, layer4]
    return ResNet(num_classes, BasicBlock, layer_params, **kwargs)


def resnet50(num_classes, **kwargs):
    layer1 = [64, 256, 1, 3]
    layer2 = [128, 512, 2, 4]
    layer3 = [256, 1024, 2, 6]
    layer4 = [512, 2048, 2, 3]
    layer_params = [layer1, layer2, layer3, layer4]
    return ResNet(num_classes, BottleNeck, layer_params, **kwargs)


def resnet101(num_classes, **kwargs):
    layer1 = [64, 256, 1, 3]
    layer2 = [128, 512, 2, 4]
    layer3 = [256, 1024, 2, 23]
    layer4 = [512, 2048, 2, 3]
    layer_params = [layer1, layer2, layer3, layer4]
    return ResNet(num_classes, BottleNeck, layer_params, **kwargs)


def resnet152(num_classes, **kwargs):
    layer1 = [64, 256, 1, 3]
    layer2 = [128, 512, 2, 8]
    layer3 = [256, 1024, 2, 36]
    layer4 = [512, 2048, 2, 3]
    layer_params = [layer1, layer2, layer3, layer4]
    return ResNet(num_classes, BottleNeck, layer_params, **kwargs)
