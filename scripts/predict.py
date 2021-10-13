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

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

    # 加载模型
    # ================================================================================
    # 方法一：加载模型源码与已保存的参数
    # model = ShuffleNetV2.shufflenet_0_1x()
    # model.load_weights(filepath=SAVED_MODEL_DIR+'model')

    # 方法二：加载已保存的模型，支持.variables, .__call__等基本属性方法的调用
    model = tf.saved_model.load(SAVED_MODEL_DIR)

    # 方法三：加载已保存的keras模型，支持.fit, .predict等方法的调用
    # model = tf.keras.models.load_model(SAVED_MODEL_DIR)

    test_file = '/tmp/whiteboard/test4.jpg'
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