import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import numpy as np
from functools import partial


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
    bboxes = tf.transpose(tf.stack((features['image/object/bbox/xmin'],
                                    features['image/object/bbox/ymin'],
                                    features['image/object/bbox/xmax'],
                                    features['image/object/bbox/ymax'])), name='bboxes')
    return image, bboxes, features['image/object/class/label'], features['image/object/class/text']


def image_argument_with_imgaug(image, bboxes, iaa_sequence=None):
    """
    增强一张图片
    :param image:   一张图片，类型为ndarray，shape为[None, None, 3]
    :param bboxes:  一组bounding box，shape 为 [bbox_number, 4]，顺序为 xmin, ymin, xmax, ymax
                        float类型，取值范围[0, 1]
    :param iaa_sequence:
    :return:        图像增强结果，包括image和bbox，其格式与输入相同
    """
    bboxes_list = []
    height, width, channels = image.shape
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(
            bbox[3] * height)
        bboxes_list.append(ia.BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax))
    bboxes_ia = ia.BoundingBoxesOnImage(bboxes_list, shape=image.shape)

    if iaa_sequence is None:
        iaa_sequence = [
            iaa.Flipud(0.5),
            iaa.Multiply((1.2, 1.5)),
            iaa.Fliplr(0.5),
            iaa.Crop(percent=(0, 0.2), keep_size=False),
            iaa.Scale({"height": 384, "width": 384}),
        ]
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


def draw_bboxes_with_labels(image, bboxes, label_texts):
    """
    在numpy对象上，画bbox和对应的labels
    :param image:       一张图片，shape 为 [height, width, channels]
    :param bboxes:      一组bounding box，shape 为 [bbox_number, 4]，顺序为 xmin, ymin, xmax, ymax
                        float类型，取值范围[0, 1]
    :param label_texts:      要显示的标签，shape为(bbox_number, )
    :return:        画完bbox的图片，为ndarray类型，shape与输入相同
    """
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    height, width, channels = image.shape
    for bbox, cur_label in zip(bboxes, label_texts):
        xmin, ymin, xmax, ymax = int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(
            bbox[3] * height)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img=image,
                    text=str(cur_label.numpy()),
                    org=(xmin, ymin + 10),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1e-3 * image.shape[0],
                    color=(0, 0, 255),
                    thickness=2
                    )
    return image


def get_dataset(tf_records_list,
                batch_size,
                shuffle=False, shuffle_buffer_size=1000,
                prefetch=False, prefetch_buffer_size=1000,
                argument=True, iaa_sequence=None):
    dataset = tf.data.TFRecordDataset(tf_records_list).map(_parse_tf_records)

    if argument:
        image_argument_partial = partial(image_argument_with_imgaug, iaa_sequence=iaa_sequence)
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
