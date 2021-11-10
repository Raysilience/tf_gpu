#!usr/bin/env python
# coding utf-8
'''
@File       :step_01_prepare_data.py
@Copyright  :CV Group
@Date       :9/29/2021
@Author     :Rui
@Desc       :
'''
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils import ShapeUtil, FileUtil
from config import *

def gen_saving_dir(root_dir, is_random=True, category="train"):
    """
    generate saving directory for every piece of data
    :param root_dir: path object of root directory
    :param is_random: boolean indicating random split or not
    :param category: one of the three options ['train', 'valid', 'test']
    :return: path object
    """
    if is_random:
        rand = random.random()
        if rand < TRAIN_SET_RATIO:
            saving_path = root_dir.joinpath('train')
        elif rand < TRAIN_SET_RATIO + VALID_SET_RATIO:
            saving_path = root_dir.joinpath('valid')
        else:
            saving_path = root_dir.joinpath('test')
    else:
        saving_path = root_dir.joinpath(category)

    return saving_path


if __name__ == '__main__':
    # data preparation pipeline
    CONVERT_RAW = False
    CONVERT_TFRECORD = True
    CLEAR_PREVIOUS_DIR = False
    RANDOM_SPLIT = True

    random.seed(0)
    # step1. convert original data into specified format under hierarchical folders
    if CONVERT_RAW:
        root = Path(RAW_DATA_DIR)
        target_root = Path(DATASET_DIR)
        LABEL_DIR = DATASET_DIR + '/label/'

        if CLEAR_PREVIOUS_DIR:
            FileUtil.clear_dir(DATASET_DIR)
        # create label directory
        FileUtil.make_dir(LABEL_DIR)

        for p in tqdm(root.glob('*.json'), desc='Processing'):
            with open(str(p), 'r') as f:
                pts = []
                data = json.load(f)
                lines = data['line']
                res = {'label': 0, 'descriptor': [0]*12}
                for line in lines:
                    if line:
                        pts.extend(line)
                pts = np.asarray(pts, dtype=np.int32)
                cls = p.stem.split('_')[0]

                folder = gen_saving_dir(target_root, is_random=RANDOM_SPLIT)

                # 处理图片
                folder = folder.joinpath(cls)
                FileUtil.make_dir(folder)
                filename = str(folder.joinpath('{0}.jpg'.format(p.stem)))
                rect = cv2.boundingRect(pts)
                x, y, w, h = rect
                img = ShapeUtil.gen_image_from_points(pts, x, y, w, h, IMAGE_WIDTH, IMAGE_HEIGHT, scale=SCALING_FACTOR)
                cv2.imwrite(filename, img)

                # 处理标签
                res['label'] = LABEL_TO_INDEX[data['label']]
                if res['label'] > 1:
                    points = data['descriptor']
                    points = ShapeUtil.map_points(points, x, y, w, h, IMAGE_WIDTH, IMAGE_HEIGHT, scale=SCALING_FACTOR).tolist()
                    for i in range(len(data['descriptor'])):
                        res['descriptor'][2*i] = points[i][0]
                        res['descriptor'][2*i+1] = points[i][1]

                with open(LABEL_DIR + p.name, 'w') as file:
                    json.dump(res, file)


    # step2. convert hierarchical data into tfrecord
    if CONVERT_TFRECORD:
        FileUtil.dataset_to_tfrecord(TRAIN_DIR, TRAIN_TFRECORD)
        FileUtil.dataset_to_tfrecord(VALID_DIR, VALID_TFRECORD)
        FileUtil.dataset_to_tfrecord(TEST_DIR, TEST_TFRECORD)

