import tensorflow as tf
import numpy as np
import cv2
import os


def get_dataset_by_local_file(mode, root_path,
                              min_edge=600, max_edge=1000):
    with open(os.path.join(root_path, 'ImageSets', 'Main', '%s.txt' % mode), 'r') as f:
        lines = f.readlines()
    examples_list = [line.strip() for line in lines]
    img_dir = os.path.join(root_path, 'JPEGImages')

    def _map_from_cv2(example):
        example = example.decode()
        img_file_path = os.path.join(img_dir, example + '.jpg')
        img = cv2.imread(img_file_path).astype(np.float32)
        img -= np.array([[[102.9801, 115.9465, 122.7717]]])
        h, w, _ = img.shape
        scale1 = min_edge / min(h, w)
        scale2 = max_edge / max(h, w)
        scale = min(scale1, scale2)
        img = cv2.resize(img, None, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_LINEAR)
        return img, float(scale), h, w

    dataset = tf.data.Dataset.from_tensor_slices(examples_list).map(
        lambda example: tf.py_func(_map_from_cv2,
                                   [example],
                                   [tf.float32, tf.float64, tf.int64, tf.int64])
    ).batch(1)

    return dataset, examples_list


def get_dataset_by_tf_records(mode, root_path,
                              min_edge=600, max_edge=1000):
    with open(os.path.join(root_path, 'ImageSets', 'Main', '%s.txt' % mode), 'r') as f:
        lines = f.readlines()
    examples_list = [line.strip() for line in lines]
    img_dir = os.path.join(root_path, 'JPEGImages')

    def _map_from_tf_image(example):
        example = example.decode()
        img_file_path = os.path.join(img_dir, example + '.jpg')
        img = tf.image.decode_jpeg(tf.io.read_file(img_file_path), channels=3)
        channels = tf.split(img, 3, axis=2)
        means = [102.9801, 115.9465, 122.7717]
        channels[0] = channels[0] - means[0]
        channels[1] = channels[1] - means[1]
        channels[2] = channels[2] - means[2]
        img = tf.concat(channels, axis=3)
        h, w, _ = img.shape
        scale1 = min_edge / min(h, w)
        scale2 = max_edge / max(h, w)
        scale = min(scale1, scale2)
        img = cv2.resize(img, None, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_LINEAR)
        img = tf.image.resize_bilinear(img, [tf.to_int32(scale*h), tf.to_int32(scale*w)])
        return img, float(scale), h, w

    dataset = tf.data.Dataset.from_tensor_slices(examples_list).map(_map_from_tf_image).batch(1)

    return dataset, examples_list
