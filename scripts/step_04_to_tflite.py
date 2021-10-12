#!usr/bin/env python
# coding utf-8
'''
@File       :step_04_to_tflite.py
@Copyright  :CV Group
@Date       :10/8/2021
@Author     :Rui
@Desc       :
'''
import numpy as np
import tensorflow as tf
from config import *
from utils.FileUtil import convert_to_tflite

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
    tflite_model_path = "{0}{1}_{2}.tflite".format(SAVED_MODEL_DIR, MODEL_NAME, VERSION)

    convert_on = True
    test_on = False

    if convert_on:
        # 转换tflite格式
        convert_to_tflite(SAVED_MODEL_DIR, tflite_model_path)

    if test_on:
        # 验证tflite模型
        interpreter = tf.lite.Interpreter(tflite_model_path)
        interpreter.allocate_tensors()

        # 获取输入输出张量
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)

        # 使用随机输入测试模型
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # 获取模型输出
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)