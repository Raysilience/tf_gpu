#!usr/bin/env python
# coding utf-8
"""
@File       :step_01_prepare_data.py
@Copyright  :CV Group
@Date       :2/9/2022
@Author     :Rui
@Desc       :
"""

import random

import numpy as np
import pandas as pd

from utils import file_util
from utils.data_loader import DataLoader
from utils.file_util import *
from utils.image_util import load_and_preprocess_image

if __name__ == "__main__":
    # LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    LOG_FORMAT = "%(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    GEN_DATA = True
    CONVERT_TFRECORD = True
    CHECK_TFRECORD = False

    if GEN_DATA:
        random.seed(1314)
        raw_data = Path(RAW_DATA_DIR)
        make_dir(TRAIN_DIR)
        make_dir(VALID_DIR)
        make_dir(TEST_DIR)
        for depth_origin in tqdm(raw_data.rglob("*Depth*"), desc='splitting'):
            tensor = load_and_preprocess_image(depth_origin)
            rand = random.random()
            if rand < TRAIN_SET_RATIO:
                target_file = TRAIN_DIR + "/" + depth_origin.stem.replace("Depth", "hand") + '.bin'
            elif rand < TRAIN_SET_RATIO + VALID_SET_RATIO:
                target_file = VALID_DIR + "/" + depth_origin.stem.replace("Depth", "hand") + '.bin'
            else:
                target_file = TEST_DIR + "/" + depth_origin.stem.replace("Depth", "hand") + '.bin'

            tensor.tofile(target_file)


    if CONVERT_TFRECORD:
        file_util.dataset_to_tfrecord(TRAIN_DIR, TRAIN_TFRECORD)
        file_util.dataset_to_tfrecord(VALID_DIR, VALID_TFRECORD)
        file_util.dataset_to_tfrecord(TEST_DIR, TEST_TFRECORD)

    if CHECK_TFRECORD:
        train_data_loader = DataLoader(TRAIN_TFRECORD)
        train_dataset = train_data_loader.get_dataset(BATCH_SIZE)
        for image_batch, label_batch in train_dataset:
            print(image_batch)
            # print(np.any(np.isnan(image_batch.numpy())))
            # print(label_batch)
            break


