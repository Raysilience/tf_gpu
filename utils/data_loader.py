#!usr/bin/env python
# coding utf-8
'''
@File       :data_loader.py
@Copyright  :CV Group
@Date       :9/27/2021
@Author     :Rui
@Desc       :
'''
import cv2
import tensorflow as tf
from scripts.config import *


class DataLoader:
    def __init__(self, tfrecord_file, labeled=True):
        self.labeled = labeled
        self.tfrecord_file = tfrecord_file
        if labeled:
            self.tfrecord_format = {
                'image': tf.io.FixedLenFeature([], tf.string),
                'label_0': tf.io.FixedLenFeature([], tf.int64),
                'label_1': tf.io.FixedLenFeature([12], tf.int64)
            }
        else:
            self.tfrecord_format = {
                'image': tf.io.FixedLenFeature([], tf.string)
            }
        self.dataset = None

    def _decode_image(self, image):
        img = tf.image.decode_jpeg(image, channels=3)
        img = tf.cast(img, tf.float32)
        img = tf.reshape(img, [IMAGE_WIDTH, IMAGE_HEIGHT, 3])
        return img

    def _read_tfrecord(self, example_string):
        """
        decoding the data from tfrecord
        :param example_string:
        :return: (image, [label])
        """
        example = tf.io.parse_single_example(example_string, self.tfrecord_format)
        image = self._decode_image(example['image'])
        if self.labeled:
            label_0 = tf.cast(example['label_0'], tf.int64)
            label_1 = tf.cast(example['label_1'], tf.int64)
            return image, label_0, label_1
        return image

    def _load_dataset(self, ordered=False):
        """
        load dataset
        :param ordered: defines whether we should read the tfrecord in order
        :return: tf.data.dataset
        """
        self.dataset = tf.data.TFRecordDataset(self.tfrecord_file)

        if not ordered:
            ignore_order = tf.data.Options()
            # disable order, increase speed
            ignore_order.experimental_deterministic = False
            # uses data as soon as it streams in, rather than in its original order
            self.dataset = self.dataset.with_options(ignore_order)

        self.dataset = self.dataset.map(self._read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self.dataset

    def get_dataset(self, batch_size=8, buffer_size=4096, prefetch=False, augment=False):
        """
        fetch dataset from recorder files
        :param batch_size:
        :param buffer_size:
        :param prefetch:
        :return: tf.data.dataset
        """
        res = self._load_dataset()
        res = res.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        if prefetch:
            res = res.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        if augment:
            x, y = list(zip(*res))
            x = tf.convert_to_tensor(x)
            aug_img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                shear_range=20,
                rotation_range=20,
                horizontal_flip=True,
                vertical_flip=True
            )
            return aug_img_gen.flow(x, y, batch_size=batch_size, shuffle=True)
        else:
            res = res.batch(batch_size=batch_size)
        return res


    def get_len(self):
        cnt = 0
        if self.dataset:
            for i in self.dataset:
                cnt += 1
        return cnt

if __name__ == '__main__':
    data_loader = DataLoader(TRAIN_TFRECORD)
    dataset = data_loader.get_dataset(
        batch_size=BATCH_SIZE,
        augment=False
    )

    cnt = 0
    for img_batch, label_0_batch, label_1_batch in dataset:
        for i in range(len(img_batch)):
            print(img_batch[i])
            print(label_0_batch[i])
            print(label_1_batch[i])

            break
        break