#!usr/bin/env python
# coding utf-8
"""
@File       :step_01_prepare_data.py
@Copyright  :CV Group
@Date       :2/9/2022
@Author     :Rui
@Desc       :
"""

import logging
import random
from pathlib import Path
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from config import *
import shutil


def load_and_preprocess_image(img_file: Path):
    """
    load image and preprocess image
    :param img_file: Path object
    :return: tensor
    """
    depth_file_name = img_file.name
    ir_file = str(img_file.parent) + "/" + depth_file_name.replace('Depth', "IR")
    # label = img_file.stem.split('_')[1]

    depth_df = pd.read_csv(str(img_file), header=None)
    ir_df = pd.read_csv(ir_file, header=None)

    depth_img = depth_df.to_numpy()
    ir_img = ir_df.to_numpy()

    return preprocess_image(depth_img, ir_img)


def preprocess_image(depth_img, ir_img):
    """
    preprocess images including scaling, stacking etc.
    :param depth_img: numpy array
    :param ir_img: numpy array
    :return: tensor
    """
    depth_img = np.where((depth_img < 3000) & (depth_img > 100), depth_img / 3000.0, 0.0)
    ir_img = np.where((ir_img > 20), ir_img / np.max(ir_img), 0.0)
    comb_tensor = np.stack([ir_img, depth_img], axis=2)
    comb_tensor = comb_tensor.astype(dtype=np.float)
    return comb_tensor

if __name__ == "__main__":
    # LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    LOG_FORMAT = "%(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    ## load file
    # data_folder = '/share/irHandData/'
    # data_root = Path(data_folder)
    # all_img_paths = list(data_root.rglob('*Depth*'))
    # print(len(all_img_paths), all_img_paths)
    # img_ds = tf.data.Dataset.from_generator(lambda: map(load_and_preprocess_image, all_img_paths),
    #                                         output_types=(tf.float32, tf.int32),
    #                                         output_shapes=((None, None, None), ()))
    # img_ds = img_ds.shuffle(200)
    # img_ds = img_ds.batch(8)
    #
    # for image_batch, label_batch in img_ds:
    #     print(label_batch)

    gen_data = False

    if gen_data:
        random.seed(1314)
        raw_data = Path(RAW_DATA_DIR)
        target_file = ''
        for depth_origin in tqdm(raw_data.rglob("*Depth*"), desc='splitting'):
            tensor = load_and_preprocess_image(depth_origin)
            rand = random.random()
            if rand < TRAIN_SET_RATIO:
                target_file = TRAIN_DIR + "/" + depth_origin.stem.replace("Depth", "hand") + '.npy'
            else:
                target_file = VALID_DIR + "/" + depth_origin.stem.replace("Depth", "hand") + '.npy'
            np.save(target_file, tensor)


    # train_data_root = Path(TRAIN_DIR)
    # all_train_img_paths = list(train_data_root.rglob('*Depth*'))
    # train_dataset = tf.data.Dataset.from_generator(lambda: map(load_and_preprocess_image, all_train_img_paths),
    #                                           output_types=(tf.float32, tf.int32),
    #                                           output_shapes=((None, None, None), ()))
    # train_dataset = train_dataset.shuffle(1024)
    # train_dataset = train_dataset.repeat()
    # train_dataset = train_dataset.batch(BATCH_SIZE)
    # train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # valid_data_root = Path(VALID_DIR)
    # all_valid_img_paths = list(train_data_root.rglob('*Depth*'))
    # valid_dataset = tf.data.Dataset.from_generator(lambda: map(load_and_preprocess_image, all_valid_img_paths),
    #                                           output_types=(tf.float32, tf.int32),
    #                                           output_shapes=((None, None, None), ()))
    # valid_dataset = valid_dataset.shuffle(256)
    # valid_dataset = valid_dataset.batch(BATCH_SIZE)
