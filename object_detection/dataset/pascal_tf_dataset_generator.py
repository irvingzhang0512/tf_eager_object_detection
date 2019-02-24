import tensorflow as tf
import numpy as np
from functools import partial

from object_detection.dataset.tf_dataset_utils import image_argument_with_imgaug, preprocessing_func

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
    features['image/object/class/text'] = tf.sparse_tensor_to_dense(features['image/object/class/text'], '')
    image = tf.image.decode_jpeg(features['image/encoded'][0])
    bboxes = tf.transpose(tf.stack((features['image/object/bbox/ymin'],
                                    features['image/object/bbox/xmin'],
                                    features['image/object/bbox/ymax'],
                                    features['image/object/bbox/xmax'])), name='bboxes')
    return image, bboxes, features['image/height'][0], features['image/width'][0], \
           features['image/object/class/label'], features['image/object/class/text']


def get_dataset(tf_records_list,
                min_size=600,
                max_size=2000,
                preprocessing_type='caffe',
                batch_size=1,
                repeat=1,
                shuffle=False, shuffle_buffer_size=1000,
                prefetch=False, prefetch_buffer_size=1000,
                argument=True, iaa_sequence=None):
    """
    获取数据集，操作过程如下：

    1) 从 tfrecords 文件中读取基本数据；
    2) 如果需要数据增强，则通过输入的 iaa_sequence 进行；
    3) 改变图片的数据格式，从 uint8 （即[0, 255]）到 float32 （即[0, 1]）
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
    1) 随机水平、垂直翻转；
    2) 随机切片

    当通过 itr 进行操作时，该 dataset 返回的数据包括：
    image, bboxes, labels, labels_text
    数据类型分别是：tf.float32([0, 1]), tf.float32([0, 边长]), tf.int32([0, num_classes]), tf.string
    shape为：[1, height, width, 3], [1, num_bboxes, 4], [num_bboxes], [num_bboxes]

    :param preprocessing_type:
    :param min_size:
    :param max_size:
    :param repeat:
    :param tf_records_list:
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
            lambda image, bboxes, image_height, image_width, labels, labels_text: tuple([
                *tf.py_func(image_argument_partial, [image, bboxes], [image.dtype, bboxes.dtype]),
                image_height, image_width, labels, labels_text])
        )

    preprocessing_partial_func = partial(preprocessing_func,
                                         min_size=min_size, max_size=max_size,
                                         preprocessing_type=preprocessing_type)

    dataset = dataset.batch(batch_size=batch_size).map(preprocessing_partial_func)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    if prefetch:
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    return dataset.repeat(repeat)


if __name__ == '__main__':
    tfs = ['/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_00.tfrecords',
           '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_01.tfrecords',
           '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_02.tfrecords',
           '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_03.tfrecords',
           '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_04.tfrecords', ]
    d = get_dataset(tfs)
    tf.enable_eager_execution()
    import matplotlib.pyplot as plt
    from object_detection.utils.visual_utils import draw_bboxes_with_labels

    for idx, (cur_image, cur_bboxes, cur_labels, cur_labels_text) in enumerate(d):
        cur_means = [103.939, 116.779, 123.68]
        cur_image = tf.squeeze(cur_image, axis=0).numpy()
        cur_image[..., 0] += cur_means[0]
        cur_image[..., 1] += cur_means[1]
        cur_image[..., 2] += cur_means[2]
        cur_image = cur_image[..., ::-1]
        cur_image = cur_image.astype(np.uint8)
        image_with_bboxes = draw_bboxes_with_labels(cur_image / 255, tf.squeeze(cur_bboxes, axis=0),
                                                    tf.squeeze(cur_labels_text, axis=0))
        plt.imshow(image_with_bboxes)
        plt.show()
        if idx == 5:
            break
