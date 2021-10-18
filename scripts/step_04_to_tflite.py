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
from utils import model_loader


def convert_to_tflite(original_saved_model_dir, export_path, quantized=False):
    """
    convert to tensorflow lite format
    :param original_saved_model_dir: string of saved model directory
    :param export_path: string of newly saved tflite model path
    :return:
    """
    # 加载模型
    model = model_loader.load(
        mode=1,
        filepath=SAVED_MODEL_DIR+'best',
        # dirpath=SAVED_MODEL_DIR
    )
    model._set_inputs(inputs=tf.TensorSpec(shape=[None, 128, 128, 3], dtype=tf.float32))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # model = tf.saved_model.load(original_saved_model_dir)
    # converter = tf.lite.TFLiteConverter.from_saved_model(model)
    if quantized:
        # 动态范围量化
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(export_path, "wb") as f:
        f.write(tflite_model)
    print('convert successfully to {}'.format(export_path))

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
    tflite_model_path = "{0}{1}_{2}.tflite".format(SAVED_MODEL_DIR, MODEL_NAME, VERSION)

    convert_on = False
    test_on = True

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