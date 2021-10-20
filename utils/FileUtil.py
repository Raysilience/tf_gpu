#!usr/bin/env python
# coding utf-8
'''
@File       :FileUtil.py
@Copyright  :CV Group
@Date       :9/26/2021
@Author     :Rui
@Desc       :
'''
import json
from pathlib import Path
import shutil
import tensorflow as tf
import logging

from tensorflow_core.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tqdm import tqdm

from scripts.config import *


def make_dir(dirname):
    p = Path(dirname)
    if not p.exists():
        p.mkdir(parents=True)


def clear_dir(dirname):
    p = Path(dirname)
    if p.exists():
        shutil.rmtree(str(p))

def gen_img_path_and_label(data_root_dir):
    """
    given root dir, generate image paths and labels correspondingly
    :param data_root_dir:
    :return: (img_path, label)
    """
    root = Path(data_root_dir)
    if not root.exists():
        raise ValueError('source directory not exist')
    img_path = list(root.glob('*/*'))
    label = [LABEL_TO_INDEX[p.stem.split('_')[0]] for p in img_path]
    return img_path, label


def gen_img_path_and_multi_label(data_root_dir):
    """
    given root dir, generate image paths and labels correspondingly
    :param data_root_dir:
    :return: (img_path, label)
    """
    root = Path(data_root_dir)
    if not root.exists():
        raise ValueError('source directory not exist')
    img_path = list(root.glob('*/*'))
    labels_0 = []
    labels_1 = []
    for path in root.glob('*/*'):
        with open(DATASET_DIR + 'label/' + path.stem + '.json', 'r') as f:
            data = json.load(f)
            labels_0.append(data['label'])
            labels_1.append(data['descriptor'])
    return img_path, labels_0, labels_1



def _build_example(image_string, labels_0, labels_1):
    """
    create a tf.train.Example from features
    :param image_string: binary representation of image
    :param labels_0: index of label in the form of int
    :param labels_1: descriptor of label in the form of int
    :return: tf.train.Example object
    """
    feature = {
        'label_0': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels_0])),
        'label_1': tf.train.Feature(int64_list=tf.train.Int64List(value=labels_1)),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string]))
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def dataset_to_tfrecord(dataset_dir, tfrecord_name):
    """
    given a dataset, save the data into a .tfrecord file
    :param dataset_dir:
    :param tfrecord_name:
    :return:
    """
    img_paths, labels_0, labels_1 = gen_img_path_and_multi_label(dataset_dir)
    with tf.io.TFRecordWriter(path=tfrecord_name) as writer:
        logging.info('Writing {} to tfrecord'.format(dataset_dir))
        for i in tqdm(range(len(img_paths)), desc='Converting'):
            with open(str(img_paths[i]), 'rb') as f:
                example = _build_example(f.read(), labels_0[i], labels_1[i])
                writer.write(example.SerializeToString())


def convert_model_to_pb(original_saved_model_dir, new_name, verbose=True):
    model = tf.keras.models.load_model(original_saved_model_dir)

    tmp = tf.function(lambda x: model(x))
    tmp = tmp.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # get frozen concrete function
    frozen_func = convert_variables_to_constants_v2(tmp)
    frozen_func.graph.as_graph_def()

    if verbose:
        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 50)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)
        print("-" * 50)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)

    # save frozen graph from frozen concrete function to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=original_saved_model_dir + 'log',
                      name=new_name + '.pb',
                      as_text=False)



if __name__ == '__main__':
    convert_model_to_pb(SAVED_MODEL_DIR, 'shufflenetv2.pb', False)
