#!usr/bin/env python
# coding utf-8
'''
@File       :ShapeUtil.py
@Copyright  :CV Group
@Date       :9/27/2021
@Author     :Rui
@Desc       :
'''
import json

import numpy as np
import cv2

def gen_image_from_points(pts, rect_x, rect_y, rect_w, rect_h, width=128, height=128, scale=1):
    """
    genenrate image from a sequence of points
    :param pts: in the form of numpy array
    :param width: output width
    :param height: output height
    :param scale: the ratio between output canvas with drawing canvas
    :return: image in the form of numpy array
    """
    assert (scale <= 1)
    x, y, w, h = rect_x, rect_y, rect_w, rect_h
    target_w = width * scale
    target_h = height * scale
    s = min(target_w / w, target_h / h)
    scaled_w = s * w
    scaled_h = s * h
    former_origin = np.array([x, y])
    latter_origin = np.array([(width - scaled_w)/2, (height - scaled_h)/2])
    new_pts = latter_origin + (pts - former_origin) * np.array([s, s])
    new_pts = new_pts.astype(np.int32)
    plane = np.zeros((width, height, 3), np.uint8)
    plane.fill(255)
    cv2.polylines(plane, [new_pts], False, (0, 0, 0), 1)
    return plane


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
