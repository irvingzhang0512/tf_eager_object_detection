import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from functools import partial

__all__ = ['get_dataset']


def _get_default_iaa_sequence():
    return [
            iaa.Flipud(0.5),
            iaa.Multiply((1.0, 1.5)),
            iaa.Fliplr(0.5),
            iaa.Crop(percent=(0, 0.2), keep_size=False),
            iaa.Scale({"height": 384, "width": 384}),
        ]


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
    features['image/object/bbox/xmin'] = tf.sparse.to_dense(features['image/object/bbox/xmin'])
    features['image/object/bbox/xmax'] = tf.sparse.to_dense(features['image/object/bbox/xmax'])
    features['image/object/bbox/ymin'] = tf.sparse.to_dense(features['image/object/bbox/ymin'])
    features['image/object/bbox/ymax'] = tf.sparse.to_dense(features['image/object/bbox/ymax'])
    features['image/object/class/label'] = tf.sparse.to_dense(features['image/object/class/label'])
    features['image/object/class/text'] = tf.sparse.to_dense(features['image/object/class/text'], '')
    image = tf.image.decode_jpeg(features['image/encoded'][0])
    bboxes = tf.transpose(tf.stack((features['image/object/bbox/ymin'],
                                    features['image/object/bbox/xmin'],
                                    features['image/object/bbox/ymax'],
                                    features['image/object/bbox/xmax'])), name='bboxes')
    return image, bboxes, features['image/object/class/label'], features['image/object/class/text']


def _image_argument_with_imgaug(image, bboxes, iaa_sequence=None):
    """
    增强一张图片
    :param image:   一张图片，类型为ndarray，shape为[None, None, 3]
    :param bboxes:  一组bounding box，shape 为 [bbox_number, 4]，顺序为 ymin, xmin, ymax, xmax
                        float类型，取值范围[0, 1]
    :param iaa_sequence:
    :return:        图像增强结果，包括image和bbox，其格式与输入相同
    """
    bboxes_list = []
    height, width, channels = image.shape
    for bbox in bboxes:
        ymin, xmin, ymax, xmax = int(bbox[0] * height), int(bbox[1] * width), int(bbox[2] * height), int(
            bbox[3] * width)
        bboxes_list.append(ia.BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax))
    bboxes_ia = ia.BoundingBoxesOnImage(bboxes_list, shape=image.shape)

    if iaa_sequence is None:
        iaa_sequence = _get_default_iaa_sequence()
    seq = iaa.Sequential(iaa_sequence)

    seq_det = seq.to_deterministic()

    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bboxes_ia])[0]

    bboxes_aug_list = []
    height, width, channels = image_aug.shape
    for iaa_bbox in bbs_aug.bounding_boxes:
        bboxes_aug_list.append([iaa_bbox.x1 / width, iaa_bbox.y1 / height, iaa_bbox.x2 / width, iaa_bbox.y2 / height])
    bboxes_aug_np = np.array(bboxes_aug_list)
    bboxes_aug_np[bboxes_aug_np < 0] = 0
    bboxes_aug_np[bboxes_aug_np > 1] = 1
    return image_aug, bboxes_aug_np.astype(np.float32)


def get_dataset(tf_records_list,
                batch_size=1,
                shuffle=False, shuffle_buffer_size=1000,
                prefetch=False, prefetch_buffer_size=1000,
                argument=True, iaa_sequence=None):
    """
    获取数据集，操作过程如下：

    1) 从 tfrecords 文件中读取基本数据；
    2) 如果需要数据增强，则通过输入的 iaa_sequence 进行；
    3) 改变图片的数据格式，从 uint8 （即[0, 255]）到 float32 （即[0, 1]）
    4) shuffle操作；
    5) prefetch操作；
    6) batch操作。

    其中，默认数据增强包括：
    ```
    iaa_sequence = [
            iaa.Flipud(0.5),
            iaa.Fliplr(0.5),
            iaa.Multiply((1.0, 1.5)),
            iaa.Crop(percent=(0, 0.2), keep_size=False),
            iaa.Scale({"height": 384, "width": 384}),
        ]
    ```
    1) 随机水平、垂直翻转；
    2) 随机亮度增强；
    3) 随机切片
    4) 将图片 resize 到 384*384

    当通过 itr 进行操作时，该 dataset 返回的数据包括：
    image, bboxes, labels, labels_text
    数据类型分别是：tf.float32([0, 1]), tf.float32([0, 1]), tf.int32([0, num_classes]), tf.string

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
        image_argument_partial = partial(_image_argument_with_imgaug, iaa_sequence=iaa_sequence)
        dataset = dataset.map(
            lambda image, bboxes, labels, labels_text: tuple([
                *tf.py_func(image_argument_partial, [image, bboxes], [image.dtype, bboxes.dtype]),
                labels, labels_text])
        )

    dataset = dataset.map(lambda image, bboxes, labels, labels_text:
                          tuple([tf.image.convert_image_dtype(image, tf.float32), bboxes, labels, labels_text]))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    if prefetch:
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    return dataset.batch(batch_size=batch_size)


