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
    img = cv2.resize(img, (128, 128))
    tensorimg = tf.convert_to_tensor(img, dtype=tf.float32)
    tensorimg = tf.expand_dims(tensorimg, axis=0)
    y0_pred, y1_pred = model(tensorimg, training=False)

    keypoints = y1_pred.numpy().reshape((6, 2)).astype(dtype=np.int)
    print(y0_pred)
    print(y1_pred)
    print(keypoints)
    for p in keypoints:
        cv2.circle(img, p, 3, (0, 255, 255), 3)
    cv2.imwrite("result.jpg", img)
    print(INDEX_TO_LABEL[np.argmax(y0_pred)])


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