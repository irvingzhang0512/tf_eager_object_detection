import tensorflow as tf


def encode_bbox_with_mean_and_std(src_bbox, dst_bbox, target_means, target_stds):
    target_means = tf.constant(target_means, dtype=tf.float32)
    target_stds = tf.constant(target_stds, dtype=tf.float32)

    box = tf.cast(src_bbox, tf.float32)
    gt_box = tf.cast(dst_bbox, tf.float32)

    width = box[..., 2] - box[..., 0] + 1.0
    height = box[..., 3] - box[..., 1] + 1.0
    center_x = box[..., 0] + 0.5 * width
    center_y = box[..., 1] + 0.5 * height

    gt_width = gt_box[..., 2] - gt_box[..., 0] + 1.0
    gt_height = gt_box[..., 3] - gt_box[..., 1] + 1.0
    gt_center_x = gt_box[..., 0] + 0.5 * gt_width
    gt_center_y = gt_box[..., 1] + 0.5 * gt_height

    dx = (gt_center_x - center_x) / width
    dy = (gt_center_y - center_y) / height
    dw = tf.log(gt_width / width)
    dh = tf.log(gt_height / height)

    delta = tf.stack([dx, dy, dw, dh], axis=-1)
    delta = (delta - target_means) / target_stds

    return delta


def decode_bbox_with_mean_and_std(anchors, bboxes_txtytwth, target_means, target_stds):
    target_means = tf.constant(
        target_means, dtype=tf.float32)
    target_stds = tf.constant(
        target_stds, dtype=tf.float32)
    delta = bboxes_txtytwth * target_stds + target_means

    # TODO fix whether to use +1 in the following two lines.
    width = anchors[:, 2] - anchors[:, 0] + 1
    height = anchors[:, 3] - anchors[:, 1] + 1
    center_x = anchors[:, 0] + 0.5 * width
    center_y = anchors[:, 1] + 0.5 * height

    center_x += delta[:, 0] * width
    center_y += delta[:, 1] * height
    width *= tf.exp(delta[:, 2])
    height *= tf.exp(delta[:, 3])

    x1 = center_x - 0.5 * width
    y1 = center_y - 0.5 * height
    x2 = x1 + width
    y2 = y1 + height
    result = tf.stack([x1, y1, x2, y2], axis=1)
    return result
