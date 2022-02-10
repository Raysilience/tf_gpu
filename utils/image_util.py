#!usr/bin/env python
# coding utf-8
'''
@File       :image_util.py
@Copyright  :CV Group
@Date       :9/27/2021
@Author     :Rui
@Desc       :
'''
import json

import numpy as np
import cv2
from pathlib import Path
import pandas as pd

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
    comb_tensor = comb_tensor.astype(dtype=np.float32)
    return comb_tensor

def gen_image_from_points(pts, rect_x, rect_y, rect_w, rect_h, width=128, height=128, scale=1):
    """
    genenrate image from a sequence of points
    :param pts: in the form of numpy array
    :param width: output width
    :param height: output height
    :param scale: the ratio between output canvas with drawing canvas
    :return: image in the form of numpy array
    """
    new_pts = map_points(pts, rect_x, rect_y, rect_w, rect_h, width, height, scale)
    plane = np.zeros((width, height, 3), np.uint8)
    plane.fill(255)
    cv2.polylines(plane, [new_pts], False, (0, 0, 0), 1)
    return plane


def map_points(pts, rect_x, rect_y, rect_w, rect_h, width=128, height=128, scale=1):
    """
    map points into input space
    :param pts: in the form of numpy array
    :param rect_x:
    :param rect_y:
    :param rect_w:
    :param rect_h:
    :param width: output width
    :param height: output height
    :param scale: the ratio between output canvas with drawing canvas
    :return: image in the form of numpy array
    """
    assert (scale <= 1)
    target_w = width * scale
    target_h = height * scale
    s = min(target_w / rect_w, target_h / rect_h)
    scaled_w = s * rect_w
    scaled_h = s * rect_h
    former_origin = np.array([rect_x, rect_y])
    latter_origin = np.array([(width - scaled_w)/2, (height - scaled_h)/2])
    new_pts = latter_origin + (pts - former_origin) * np.array([s, s])
    new_pts = new_pts.astype(np.int32)
    return new_pts

if __name__ == '__main__':
    with open('/tmp/whiteboard/raw_data/quadrangle_20210907151719.json', 'r') as f:
        pts = []
        lines = json.load(f)['line']
        for line in lines:
            if line:
                pts.extend(line)
        pts = np.asarray(pts, dtype=np.int32)
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        img = gen_image_from_points(pts, x, y, w, h, 128, 128, scale=0.8)
        cv2.imwrite('/tmp/whiteboard/shapeUtil_test.jpg', img)
