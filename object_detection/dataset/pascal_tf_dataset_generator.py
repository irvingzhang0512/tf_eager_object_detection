import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from functools import partial

__all__ = ['get_dataset']


def _get_default_iaa_sequence():
    return [
        iaa.Flipud(0.5),
        iaa.Fliplr(0.5),
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
    return image, bboxes, features['image/height'][0], features['image/width'][0], features['image/object/class/label'], \
           features['image/object/class/text']


def _image_argument_with_imgaug(image, bboxes, iaa_sequence=None):
    """
    增强一张图片
    输入图像是 tf.uint8 类型，数据范围 [0, 255]
    输入bboxes是 tf.float32 类型，数据范围 [0, 1]
    返回结果与输入相同
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
        bboxes_aug_list.append([iaa_bbox.y1 / height, iaa_bbox.x1 / width, iaa_bbox.y2 / height, iaa_bbox.x2 / width])
    bboxes_aug_np = np.array(bboxes_aug_list)
    bboxes_aug_np[bboxes_aug_np < 0] = 0
    bboxes_aug_np[bboxes_aug_np > 1] = 1
    return image_aug, bboxes_aug_np.astype(np.float32)


def _caffe_preprocessing(image):
    """
    输入 uint8 RGB 的图像，转换为 tf.float32 BGR 格式，并减去 imagenet 平均数
    :param image:
    :return:
    """
    image = tf.to_float(image)
    image = image[..., ::-1]
    means = [103.939, 116.779, 123.68]
    channels = tf.split(axis=-1, num_or_size_splits=3, value=image)
    for i in range(3):
        channels[i] -= means[i]
    return tf.concat(axis=-1, values=channels)


def _tf_preprocessing(image):
    """
    输入 uint8 RGB 的图像，转换为 tf.float32 RGB 格式，取值范围[0, 1]
    :param image:
    :return:
    """
    return tf.image.convert_image_dtype(image, tf.float32)


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
            iaa.Flipud(0.5),
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
        image_argument_partial = partial(_image_argument_with_imgaug, iaa_sequence=iaa_sequence)
        dataset = dataset.map(
            lambda image, bboxes, image_height, image_width, labels, labels_text: tuple([
                *tf.py_func(image_argument_partial, [image, bboxes], [image.dtype, bboxes.dtype]),
                image_height, image_width, labels, labels_text])
        )

    def _map_after_batch(image, bboxes, height, width, labels, labels_text):
        """
        rescale image
        1) 短边最短为600，长边最长为2000，矛盾时，优先满足长边2000
        2) 输入数据bboxes，本来是[0, 1]， 转换为像素值
        3) 通过 preprocessing_type 选择 preprocessing 函数
        :param image: 
        :param bboxes: 
        :param labels: 
        :param labels_text: 
        :return: 
        """
        height = tf.to_float(height[0])
        width = tf.to_float(width[0])
        scale1 = min_size / tf.minimum(height, width)
        scale2 = max_size / tf.minimum(height, width)
        scale = tf.minimum(scale1, scale2)
        n_height = scale * height
        n_width = scale * width

        channels = tf.split(axis=-1, num_or_size_splits=4, value=bboxes)
        channels[0] = channels[0] * n_height
        channels[1] = channels[1] * n_width
        channels[2] = channels[2] * n_height
        channels[3] = channels[3] * n_width
        bboxes = tf.concat(channels, axis=-1)

        image = tf.image.resize_bilinear(image, (tf.to_int32(n_height), tf.to_int32(n_width)))

        if preprocessing_type == 'caffe':
            preprocessing_fn = _caffe_preprocessing
        elif preprocessing_type == 'tf':
            preprocessing_fn = _tf_preprocessing
        else:
            raise ValueError('unknown preprocessing type {}'.format(preprocessing_type))

        return preprocessing_fn(image), bboxes, labels, labels_text

    dataset = dataset.batch(batch_size=batch_size).map(_map_after_batch)

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
        print(image_with_bboxes.shape)
        plt.show()
        if idx == 5:
            break
