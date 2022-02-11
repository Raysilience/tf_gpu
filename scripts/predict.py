#!usr/bin/env python
# coding utf-8
'''
@File       :predict.py
@Copyright  :CV Group
@Date       :9/30/2021
@Author     :Rui
@Desc       :
'''
from pathlib import Path
import tensorflow as tf
import cv2
import numpy as np
from config import *
from models import ShuffleNetV2
from utils import model_loader
from utils.image_util import load_and_preprocess_image


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus[3], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[3], enable=True)

    # 加载模型
    model = model_loader.load(
        mode=1,
        model_name='MobileNetV2',
        filepath=SAVED_MODEL_DIR+'best',
        dirpath=SAVED_MODEL_DIR
    )

    test_root = Path("/share/irHandData/handtest")
    for depth_origin in test_root.rglob('*Depth*'):
        tensor = load_and_preprocess_image(depth_origin)
        tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
        tensor = tf.expand_dims(tensor, axis=0)
        pred = model(tensor, training=False)
        print("Model prediction of {} is {} with probability {}".format(depth_origin.stem,
                                                                        np.argmax(pred),
                                                                        pred))

