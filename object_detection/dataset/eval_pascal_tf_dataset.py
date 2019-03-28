import tensorflow as tf
import numpy as np
import cv2
import os
from functools import partial


__all__ = ['get_dataset_by_tf_records', 'get_dataset_by_local_file']


def get_dataset_by_local_file(mode, root_path, image_format='bgr',
                              preprocessing_type='caffe', caffe_pixel_means=None,
                              min_edge=600, max_edge=1000):
    """
    根据 /path/to/VOC2007 or VOC2012/ImageSets/Main/{}.txt 读取图片列表，读取图片
    :param mode:
    :param root_path:
    :param image_format:
    :param caffe_pixel_means: 
    :param preprocessing_type:
    :param min_edge: 
    :param max_edge: 
    :return: 
    """
    if image_format not in ['rgb', 'bgr']:
        raise ValueError('unknown image format {}'.format(image_format))
    with open(os.path.join(root_path, 'ImageSets', 'Main', '%s.txt' % mode), 'r') as f:
        lines = f.readlines()
    examples_list = [line.strip() for line in lines]
    img_dir = os.path.join(root_path, 'JPEGImages')

    def _map_from_cv2(example):
        example = example.decode()
        img_file_path = os.path.join(img_dir, example + '.jpg')
        img = cv2.imread(img_file_path).astype(np.float32)
        if preprocessing_type == 'caffe':
            img -= np.array([[caffe_pixel_means]])
        elif preprocessing_type == 'tf':
            img = img / 255.0 * 2.0 - 1.0
        else:
            raise ValueError('unknown preprocessing type {}'.format(preprocessing_type))
        h, w, _ = img.shape
        scale1 = min_edge / min(h, w)
        scale2 = max_edge / max(h, w)
        scale = min(scale1, scale2)
        new_h = int(scale * h)
        new_w = int(scale * w)

        img = cv2.resize(img, (new_w, new_h))
        if image_format == 'rgb':
            img = img[..., ::-1]
        return img, float(scale), h, w

    dataset = tf.data.Dataset.from_tensor_slices(examples_list).map(
        lambda example: tf.py_func(_map_from_cv2,
                                   [example],
                                   [tf.float32, tf.float64, tf.int64, tf.int64]  # linux
                                   # [tf.float32, tf.float64, tf.int32, tf.int32]  # windows
                                   )
    ).batch(1)

    return dataset, examples_list


def _caffe_preprocessing(image, pixel_means):
    """
    输入 uint8 RGB 的图像，转换为 tf.float32 BGR 格式，并减去 imagenet 平均数
    :param image:
    :return:
    """
    image = tf.to_float(image)
    image = tf.reverse(image, axis=[-1])
    channels = tf.split(axis=-1, num_or_size_splits=3, value=image)
    for i in range(3):
        channels[i] -= pixel_means[i]
    return tf.concat(axis=-1, values=channels)


def _tf_preprocessing(image):
    """
    输入 uint8 RGB 的图像，转换为 tf.float32 RGB 格式，取值范围[-1, 1]
    :param image:
    :return:
    """
    return tf.image.convert_image_dtype(image, dtype=tf.float32) * 2.0 - 1.0


def get_dataset_by_tf_records(mode, root_path,
                              preprocessing_type='caffe', caffe_pixel_means=None,
                              min_edge=600, max_edge=1000):
    with open(os.path.join(root_path, 'ImageSets', 'Main', '%s.txt' % mode), 'r') as f:
        lines = f.readlines()
    examples_list = [line.strip() for line in lines]
    img_dir = os.path.join(root_path, 'JPEGImages')
    example_path_list = [os.path.join(img_dir, example+'.jpg') for example in examples_list]

    def _map_from_tf_image(example_path):
        img = tf.image.decode_jpeg(tf.io.read_file(example_path), channels=3)
        if preprocessing_type == 'caffe':
            preprocessing_fn = partial(_caffe_preprocessing, pixel_means=caffe_pixel_means)
        elif preprocessing_type == 'tf':
            preprocessing_fn = _tf_preprocessing
        else:
            raise ValueError('unknown preprocessing type {}'.format(preprocessing_type))
        img = preprocessing_fn(img)

        # TODO: could not get image shape
        h, w, _ = img.get_shape().as_list()
        scale1 = min_edge / min(h, w)
        scale2 = max_edge / max(h, w)
        scale = min(scale1, scale2)
        img = tf.image.resize_bilinear(img, [tf.to_int32(scale*h), tf.to_int32(scale*w)])
        return img, float(scale), h, w

    dataset = tf.data.Dataset.from_tensor_slices(example_path_list).map(_map_from_tf_image).batch(1)

    return dataset, examples_list
