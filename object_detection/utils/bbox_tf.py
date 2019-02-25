import tensorflow as tf


__all__ = ['pairwise_iou', 'bboxes_clip_filter', 'bboxes_range_filter']


def area(boxes):
    """
    Args:
      boxes: nx4 floatbox
    Returns:
      n
    """
    x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
    return tf.squeeze((y_max - y_min + 1.0) * (x_max - x_min + 1.0), [1])


def pairwise_intersection(boxlist1, boxlist2):
    """Compute pairwise intersection areas between boxes.
    Args:
      boxlist1: Nx4 floatbox
      boxlist2: Mx4
    Returns:
      a tensor with shape [N, M] representing pairwise intersections
    """
    x_min1, y_min1, x_max1, y_max1 = tf.split(boxlist1, 4, axis=1)
    x_min2, y_min2, x_max2, y_max2 = tf.split(boxlist2, 4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin + 1.0)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin + 1.0)
    return intersect_heights * intersect_widths


def pairwise_iou(boxlist1, boxlist2):
    """Computes pairwise intersection-over-union between box collections.
    copy from https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/utils/box_ops.py
    Args:
      boxlist1: Nx4 floatbox
      boxlist2: Mx4
    Returns:
      a tensor with shape [N, M] representing pairwise iou scores.
    """
    boxlist1 = tf.to_float(boxlist1)
    boxlist2 = tf.to_float(boxlist2)

    intersections = pairwise_intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))


def bboxes_clip_filter(rpn_proposals, min_value, max_height, max_width, min_edge=None):
    """
    numpy 操作
    根据边界、最小边长过滤 proposals
    :param rpn_proposals:           bboxes
    :param min_value:
    :param max_height:
    :param max_width:
    :param min_edge:
    :return:
    """
    rpn_proposals = tf.where(rpn_proposals < min_value, tf.ones_like(rpn_proposals) * min_value, rpn_proposals)

    channels = tf.split(rpn_proposals, 4, axis=1)
    channels[0] = tf.maximum(tf.minimum(channels[0], max_height - 1), min_value)
    channels[1] = tf.maximum(tf.minimum(channels[1], max_width - 1), min_value)
    channels[2] = tf.maximum(tf.minimum(channels[2], max_height - 1), min_value)
    channels[3] = tf.maximum(tf.minimum(channels[3], max_width - 1), min_value)
    # channels[0] = tf.where(channels[0] > max_height, tf.ones_like(channels[0]) * max_height, channels[0])
    # channels[1] = tf.where(channels[1] > max_width, tf.ones_like(channels[1]) * max_width, channels[1])
    # channels[2] = tf.where(channels[2] > max_height, tf.ones_like(channels[2]) * max_height, channels[2])
    # channels[3] = tf.where(channels[3] > max_width, tf.ones_like(channels[3]) * max_width, channels[3])

    if min_edge is None:
        rpn_proposals = tf.concat(channels, axis=1)
        return rpn_proposals, tf.range(rpn_proposals.shape[0])

    min_edge = tf.to_float(min_edge)
    y_len = tf.to_float(channels[2] - channels[0] + 1.0)
    x_len = tf.to_float(channels[3] - channels[1] + 1.0)
    rpn_proposals_idx = tf.where(tf.logical_and(x_len >= min_edge, y_len >= min_edge))
    rpn_proposals_idx = rpn_proposals_idx[:, 0]
    return tf.gather(rpn_proposals, rpn_proposals_idx), rpn_proposals_idx


def bboxes_range_filter(anchors, max_height, max_width):
    """
    过滤 anchors，超出图像范围的 anchors 都不要
    :param anchors:
    :param max_height:
    :param max_width:
    :return:
    """
    index_inside = tf.where(
        tf.logical_and(
            tf.logical_and((anchors[:, 0] >= 0), (anchors[:, 1] >= 0)),
            tf.logical_and((anchors[:, 2] <= max_height - 1), (anchors[:, 3] <= max_width - 1)),
        )
    )[:, 0]
    return index_inside
