#!usr/bin/env python
# coding utf-8
'''
@File       :model_loader.py
@Copyright  :CV Group
@Date       :10/14/2021
@Author     :Rui
@Desc       :
'''
from models import ShuffleNetV2
from models.CNN import CNN

from scripts.config import *
import tensorflow as tf


def load(mode, filepath=None, dirpath=None):
    """
    load model from previously saved model or weights
    :param mode: 0 means loading a newly build model, 1 means loading from saved weights, 2 means loading from saved model
    :param filepath: string of saved weights
    :param dirpath: string of saved model directory
    :return: keras model object
    """
    if mode == 0:
        # model = CNN()
        model = ShuffleNetV2.shufflenet_0_1x()

    # load model from saved weights
    elif mode == 1:
        # model = CNN()
        model = ShuffleNetV2.shufflenet_0_1x()
        model.load_weights(filepath=filepath)

    # load model from saved model
    else:
        # model = tf.saved_model.load(dirpath)
        model = tf.keras.models.load_model(dirpath)

    return model

