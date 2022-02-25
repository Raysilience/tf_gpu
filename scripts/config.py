#!usr/bin/env python
# coding utf-8
"""
@File       :config.py
@Copyright  :CV Group
@Date       :2/9/2022
@Author     :Rui
@Desc       :
"""

DEVICE = 'GPU'
MODEL_NAME = 'irHand'
VERSION = 'V1_0_0'

# training params
NUM_EPOCHS = 300
BATCH_SIZE = 64
NUM_CLASSES = 5

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
CHANNELS = 2

# model saving directory
SAVED_MODEL_DIR = '/tmp/irHand/saved_model/'
SAVE_EVERY_N_EPOCH = 30

# training summary directory
SUMMARY_DIR = '/tmp/irHand/tensorboard'

# dataset directories
RAW_DATA_DIR = '/share/irHandData/raw'
DATASET_DIR = '/share/irHandData/'
TRAIN_DIR = DATASET_DIR + 'train'
VALID_DIR = DATASET_DIR + 'valid'
TEST_DIR = DATASET_DIR + 'test'

TRAIN_TFRECORD = DATASET_DIR + 'train.tfrecord'
VALID_TFRECORD = DATASET_DIR + 'valid.tfrecord'
TEST_TFRECORD = DATASET_DIR + 'test.tfrecord'

TRAIN_SET_RATIO = 0.7
TEST_SET_RATIO = 0.1
VALID_SET_RATIO = 0.2

# label-index conversion
LABEL_TO_INDEX = {
    'unknown': 0,
    'ellipse': 1,
    'triangle': 2,
    'quadrangle': 3,
    'pentagon': 4,
    'hexagon': 5,
}

INDEX_TO_LABEL = {
    0: 'unknown',
    1: 'ellipse',
    2: 'triangle',
    3: 'quadrangle',
    4: 'pentagon',
    5: 'hexagon',
}
