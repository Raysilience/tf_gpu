#!usr/bin/env python
# coding utf-8
'''
@File       :step_03_evaluate.py
@Copyright  :CV Group
@Date       :9/28/2021
@Author     :Rui
@Desc       :
'''
import tensorflow as tf
from tqdm import tqdm

from utils import model_loader
from utils.data_loader import DataLoader
from scripts.config import *

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

    # 加载模型
    model = model_loader.load(
        mode=1,
        model_name='ResNet50',
        filepath=SAVED_MODEL_DIR+'best',
        # dirpath=SAVED_MODEL_DIR
    )

    data_loader = DataLoader(TEST_TFRECORD)
    dataset = data_loader.get_dataset()

    # 模型评估
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    for x_test, y0_test in tqdm(dataset, desc='evaluating test performance'):
        y0_pred = model(x_test, training=False)
        test_acc.update_state(y_true=y0_test, y_pred=y0_pred)

    print("test accuracy: {:.5f}".format(test_acc.result()))
