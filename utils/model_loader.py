#!usr/bin/env python
# coding utf-8
'''
@File       :model_loader.py
@Copyright  :CV Group
@Date       :10/14/2021
@Author     :Rui
@Desc       :
'''
from models import ShuffleNetV2, MobileNetV2, MobileNetV1, ResNet

from scripts.config import *
import tensorflow as tf


def _select_model(model):
    if model is None:
        raise RuntimeError('cannot load none object')
    elif model == 'ShuffleNet_0_1x':
        return ShuffleNetV2.shufflenet_0_1x()
    elif model == 'ShuffleNet_0_5x':
        return ShuffleNetV2.shufflenet_0_5x()
    elif model == 'ShuffleNet_1_0x':
        return ShuffleNetV2.shufflenet_1_0x()
    elif model == 'ShuffleNet_1_5x':
        return ShuffleNetV2.shufflenet_1_5x()
    elif model == 'ShuffleNet_2_0x':
        return ShuffleNetV2.shufflenet_2_0x()

    elif model == 'MobileNetV1_0_5x':
        return MobileNetV1.MobileNetV1(alpha=0.5)
    elif model == 'MobileNetV1_0_75x':
        return MobileNetV1.MobileNetV1(alpha=0.75)
    elif model == 'MobileNetV1_1_0x':
        return MobileNetV1.MobileNetV1()
    elif model == 'MobileNetV2':
        return MobileNetV2.MobileNetV2()

    elif model == 'ResNet18':
        return ResNet.resnet18(NUM_CLASSES)
    elif model == 'ResNet34':
        return ResNet.resnet34(NUM_CLASSES)
    elif model == 'ResNet50':
        return ResNet.resnet50(NUM_CLASSES)
    elif model == 'ResNet101':
        return ResNet.resnet101(NUM_CLASSES)
    elif model == 'ResNet152':
        return ResNet.resnet152(NUM_CLASSES)

    else:
        raise ValueError(model + ' is not support yet')


def load(mode, model_name=None, filepath=None, dirpath=None):
    """
    load model from previously saved model or weights
    :param mode: 0 means loading a newly build model, 1 means loading from saved weights, 2 means loading from saved model
    :param filepath: string of saved weights
    :param dirpath: string of saved model directory
    :return: keras model object
    """
    if mode == 0:
        model = _select_model(model_name)

    # load model from saved weights
    elif mode == 1:
        model = _select_model(model_name)
        model.load_weights(filepath=filepath)

    # load model from saved model
    elif mode == 2:
        # model = tf.saved_model.load(dirpath)
        model = tf.keras.models.load_model(dirpath)

    else:
        raise ValueError("mode not implemented")


    return model

