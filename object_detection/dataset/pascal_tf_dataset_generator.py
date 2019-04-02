import tensorflow as tf
from functools import partial

from object_detection.dataset.utils.tf_dataset_utils import image_argument_with_imgaug, preprocessing_training_func

__all__ = ['get_dataset']


def _parse_tf_records(serialized_example):
    features = tf.parse_single_example(serialized_example,
                                       features={'image/height': tf.FixedLenFeature([1], tf.int64),
                                                 'image/width': tf.FixedLenFeature([1], tf.int64),
                                                 'image/filename': tf.FixedLenFeature([1], tf.string),
                                                 'image/encoded': tf.FixedLenFeature([1], tf.string),
                                                 'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
                                                 'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
                                                 'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
                                                 'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
                                                 'image/object/class/label': tf.VarLenFeature(tf.int64),
                                                 'image/object/class/text': tf.VarLenFeature(tf.string),
                                                 }
                                       )
    features['image/object/bbox/xmin'] = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'])
    features['image/object/bbox/xmax'] = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax'])
    features['image/object/bbox/ymin'] = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'])
    features['image/object/bbox/ymax'] = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax'])
    features['image/object/class/label'] = tf.sparse_tensor_to_dense(features['image/object/class/label'])
    image = tf.image.decode_jpeg(features['image/encoded'][0])
    bboxes = tf.transpose(tf.stack((features['image/object/bbox/ymin'],
                                    features['image/object/bbox/xmin'],
                                    features['image/object/bbox/ymax'],
                                    features['image/object/bbox/xmax'])), name='bboxes')
    return image, bboxes, features['image/height'][0], features['image/width'][0], features['image/object/class/label']


def get_dataset(tf_records_list,
                min_size=600, max_size=1000,
                preprocessing_type='caffe', caffe_pixel_means=None,
                batch_size=1, repeat=1,
                shuffle=False, shuffle_buffer_size=1000,
                prefetch=False, prefetch_buffer_size=1000,
                argument=True, iaa_sequence=None):
    """
    获取数据集，操作过程如下：

    1) 从 tfrecords 文件中读取基本数据；
    2) 如果需要数据增强，则通过输入的 iaa_sequence 进行；
    3) 数据归一化，将 uint8 转换为 float，可能是转换到[0, 1]之间，也可能是减去像素平均数
    4) shuffle 操作；
    5) prefetch 操作；
    6) batch 操作。
    7) repeat 操作

    其中，默认数据增强包括：
    ```
    iaa_sequence = [
            iaa.Fliplr(0.5),
        ]
    ```
    1) 随机水平；

    当通过 itr 进行操作时，该 dataset 返回的数据包括：
    image, bboxes, labels
    数据类型分别是：tf.float32([0, 1]), tf.float32([0, 边长]), tf.int32([0, num_classes])
    shape为：[1, height, width, 3], [1, num_bboxes, 4], [num_bboxes]

    :param tf_records_list:
    :param min_size:
    :param max_size:
    :param preprocessing_type:
    :param caffe_pixel_means:
    :param repeat:
    :param batch_size:
    :param shuffle:
    :param shuffle_buffer_size:
    :param prefetch:
    :param prefetch_buffer_size:
    :param argument:
    :param iaa_sequence:
    :return:
    """

    dataset = tf.data.TFRecordDataset(tf_records_list).map(_parse_tf_records)

    if argument:
        image_argument_partial = partial(image_argument_with_imgaug, iaa_sequence=iaa_sequence)
        dataset = dataset.map(
            lambda image, bboxes, image_height, image_width, labels: tuple([
                *tf.py_func(image_argument_partial, [image, bboxes], [image.dtype, bboxes.dtype]),
                image_height, image_width, labels])
        )

    preprocessing_partial_func = partial(preprocessing_training_func,
                                         min_size=min_size, max_size=max_size,
                                         preprocessing_type=preprocessing_type,
                                         caffe_pixel_means=caffe_pixel_means)

    dataset = dataset.batch(batch_size=batch_size).map(preprocessing_partial_func)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    if prefetch:
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    return dataset.repeat(repeat)
