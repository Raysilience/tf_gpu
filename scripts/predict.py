#!usr/bin/env python
# coding utf-8
'''
@File       :predict.py
@Copyright  :CV Group
@Date       :9/30/2021
@Author     :Rui
@Desc       :
'''
import tensorflow as tf
import cv2
import numpy as np
from config import *
from models import ShuffleNetV2
from utils import model_loader

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

    # 加载模型
    model = model_loader.load(
        mode=1,
        filepath=SAVED_MODEL_DIR+'best',
        # dirpath=SAVED_MODEL_DIR
    )

    test_file = '/tmp/whiteboard/test5.jpg'
    img = cv2.imread(test_file)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)
    pred = model(img, training=False)


    print(pred)
    print(INDEX_TO_LABEL[np.argmax(pred)])


    # 验证数据增强
    # aug_img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    #     shear_range=20
    #     # rotation_range=180
    #     # horizontal_flip=True
    # )
    #
    # aug_iter = aug_img_gen.flow(img)
    # for i in range(8):
    #     new_img = next(aug_iter)
    #     cv2.imwrite('/tmp/whiteboard/aug_test_{}.jpg'.format(i), new_img[0])