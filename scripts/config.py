#!usr/bin/env python
# coding utf-8
'''
@File       :config.py.py
@Copyright  :CV Group
@Date       :9/26/2021
@Author     :Rui
@Desc       :
'''

DEVICE = 'gpu'
MODEL_NAME = 'risc'
VERSION = 'V1_0_0'

# training params
NUM_EPOCHS = 300
BATCH_SIZE = 32
NUM_CLASSES = 6

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
CHANNELS = 3
SCALING_FACTOR = 0.7

SAVED_MODEL_DIR = '/tmp/whiteboard/saved_model/'
SAVE_EVERY_N_EPOCH = 100
EXPORT_METADATA_DIR = SAVED_MODEL_DIR + 'meta/'
LABEL_FILE = '/tmp/whiteboard/label.txt'

RAW_DATA_DIR = '/tmp/whiteboard/raw_data/'
DATASET_DIR = '/tmp/whiteboard/dataset/'
TRAIN_DIR = DATASET_DIR + 'train'
VALID_DIR = DATASET_DIR + 'valid'
TEST_DIR = DATASET_DIR + 'test'

TRAIN_TFRECORD = DATASET_DIR + 'train.tfrecord'
VALID_TFRECORD = DATASET_DIR + 'valid.tfrecord'
TEST_TFRECORD = DATASET_DIR + 'test.tfrecord'

TRAIN_SET_RATIO = 0.7
TEST_SET_RATIO = 0.1
VALID_SET_RATIO = 0.2

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