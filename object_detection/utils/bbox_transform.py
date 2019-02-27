import numpy as np
import tensorflow as tf


# def encode_bbox_with_mean_and_std(src_bbox, dst_bbox, target_means, target_stds):
#     target_means = tf.constant(target_means, dtype=tf.float32)
#     target_stds = tf.constant(target_stds, dtype=tf.float32)
#
#     box = tf.cast(src_bbox, tf.float32)
#     gt_box = tf.cast(dst_bbox, tf.float32)
#
#     width = box[..., 2] - box[..., 0] + 1.0
#     height = box[..., 3] - box[..., 1] + 1.0
#     center_x = box[..., 0] + 0.5 * width
#     center_y = box[..., 1] + 0.5 * height
#
#     gt_width = gt_box[..., 2] - gt_box[..., 0] + 1.0
#     gt_height = gt_box[..., 3] - gt_box[..., 1] + 1.0
#     gt_center_x = gt_box[..., 0] + 0.5 * gt_width
#     gt_center_y = gt_box[..., 1] + 0.5 * gt_height
#
#     dy = (gt_center_y - center_y) / height
#     dx = (gt_center_x - center_x) / width
#     dh = tf.log(gt_height / height)
#     dw = tf.log(gt_width / width)
#
#     delta = tf.stack([dx, dy, dw, dh], axis=-1)
#     delta = (delta - target_means) / target_stds
#
#     return delta
#
#
# def decode_bbox_with_mean_and_std(anchors, bboxes_txtytwth, target_means, target_stds):
#     target_means = tf.constant(
#         target_means, dtype=tf.float32)
#     target_stds = tf.constant(
#         target_stds, dtype=tf.float32)
#     delta = bboxes_txtytwth * target_stds + target_means
#
#     width = anchors[:, 2] - anchors[:, 0] + 1.0
#     height = anchors[:, 3] - anchors[:, 1] + 1.0
#     center_x = anchors[:, 0] + 0.5 * width
#     center_y = anchors[:, 1] + 0.5 * height
#
#     center_x += delta[:, 0] * width
#     center_y += delta[:, 1] * height
#     width *= tf.exp(delta[:, 2])
#     height *= tf.exp(delta[:, 3])
#
#     x1 = center_x - 0.5 * width
#     y1 = center_y - 0.5 * height
#     x2 = x1 + width
#     y2 = y1 + height
#     result = tf.stack([x1, y1, x2, y2], axis=1)
#     return result


def encode_bbox_with_mean_and_std(src_bbox, dst_bbox, target_means, target_stds):
    """Compute refinement needed to transform box to gt_box.

    Args
    ---
        src_bbox: [..., (y1, x1, y2, x2)]
        dst_bbox: [..., (y1, x1, y2, x2)]
        target_means: [4]
        target_stds: [4]
    """
    target_means = tf.constant(target_means, dtype=tf.float32)
    target_stds = tf.constant(target_stds, dtype=tf.float32)

    box = tf.cast(src_bbox, tf.float32)
    gt_box = tf.cast(dst_bbox, tf.float32)

    height = box[..., 2] - box[..., 0] + 1.0
    width = box[..., 3] - box[..., 1] + 1.0
    center_y = box[..., 0] + 0.5 * height
    center_x = box[..., 1] + 0.5 * width

    gt_height = gt_box[..., 2] - gt_box[..., 0] + 1.0
    gt_width = gt_box[..., 3] - gt_box[..., 1] + 1.0
    gt_center_y = gt_box[..., 0] + 0.5 * gt_height
    gt_center_x = gt_box[..., 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    delta = tf.stack([dy, dx, dh, dw], axis=-1)
    delta = (delta - target_means) / target_stds

    return delta


def decode_bbox_with_mean_and_std(anchors, bboxes_txtytwth, target_means, target_stds):
    """Compute bounding box based on roi and delta.

    Args
    ---
        anchors: [N, (y1, x1, y2, x2)] box to update
        bboxes_txtytwth: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        target_means: [4]
        target_stds: [4]
    """
    target_means = tf.constant(
        target_means, dtype=tf.float32)
    target_stds = tf.constant(
        target_stds, dtype=tf.float32)
    delta = bboxes_txtytwth * target_stds + target_means
    # Convert to y, x, h, w
    height = anchors[:, 2] - anchors[:, 0] + 1.0
    width = anchors[:, 3] - anchors[:, 1] + 1.0
    center_y = anchors[:, 0] + 0.5 * height
    center_x = anchors[:, 1] + 0.5 * width

    # Apply delta
    center_y += delta[:, 0] * height
    center_x += delta[:, 1] * width
    height *= tf.exp(delta[:, 2])
    width *= tf.exp(delta[:, 3])

    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1)
    return result


def encode_bbox_np(src_bbox, dst_bbox):
    """
    https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/model/utils/bbox_tools.py
    :param src_bbox: (N, 4), ymin xmin ymax xmax
    :param dst_bbox: (N, 4), ymin xmin ymax xmax
    :return:
    """
    height = src_bbox[:, 2] - src_bbox[:, 0] + 1.0
    width = src_bbox[:, 3] - src_bbox[:, 1] + 1.0
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0] + 1.0
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1] + 1.0
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc


def decode_bbox_np(anchors, bboxes_txtytwth):
    """
    https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/model/utils/bbox_tools.py
    :param anchors:  (N, 4), ymin xmin ymax xmax
    :param bboxes_txtytwth:  (N, 4), dy dx dh dw
    :return:                ymin xmin ymax xmax
    """
    src_bbox = anchors.astype(anchors.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0] + 1.0
    src_width = src_bbox[:, 3] - src_bbox[:, 1] + 1.0
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = bboxes_txtytwth[:, 0]
    dx = bboxes_txtytwth[:, 1]
    dh = bboxes_txtytwth[:, 2]
    dw = bboxes_txtytwth[:, 3]

    ctr_y = dy * src_height + src_ctr_y
    ctr_x = dx * src_width + src_ctr_x
    h = np.exp(dh) * src_height
    w = np.exp(dw) * src_width

    dst_bbox = np.zeros(src_bbox.shape, dtype=src_height.dtype)
    dst_bbox[:, 0] = ctr_y - 0.5 * h
    dst_bbox[:, 1] = ctr_x - 0.5 * w
    dst_bbox[:, 2] = ctr_y + 0.5 * h
    dst_bbox[:, 3] = ctr_x + 0.5 * w
    return dst_bbox
