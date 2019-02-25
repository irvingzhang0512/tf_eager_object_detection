# copy from https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/utils/np_box_ops.py
import numpy as np


__all__ = ['pairwise_iou', 'ioa', 'bboxes_clip_filter', 'bboxes_range_filter']


def area(boxes):
    """Computes area of boxes.
    Args:
      boxes: Numpy array with shape [N, 4] holding N boxes
    Returns:
      a numpy array with shape [N*1] representing box areas
    """
    return (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)


def intersection(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.
    Args:
      boxes1: a numpy array with shape [N, 4] holding N boxes
      boxes2: a numpy array with shape [M, 4] holding M boxes
    Returns:
      a numpy array with shape [N*M] representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape, dtype='f4'),
        all_pairs_min_ymax - all_pairs_max_ymin + 1)
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape, dtype='f4'),
        all_pairs_min_xmax - all_pairs_max_xmin + 1)
    return intersect_heights * intersect_widths


def pairwise_iou(boxes1, boxes2):
    """Computes pairwise intersection-over-union between box collections.
    Args:
      boxes1: a numpy array with shape [N, 4] holding N boxes.
      boxes2: a numpy array with shape [M, 4] holding M boxes.
    Returns:
      a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = intersection(boxes1, boxes2)
    area1 = area(boxes1)
    area2 = area(boxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(
        area2, axis=0) - intersect
    return intersect / union


def ioa(boxes1, boxes2):
    """Computes pairwise intersection-over-area between box collections.
    Intersection-over-area (ioa) between two boxes box1 and box2 is defined as
    their intersection area over box2's area. Note that ioa is not symmetric,
    that is, IOA(box1, box2) != IOA(box2, box1).
    Args:
      boxes1: a numpy array with shape [N, 4] holding N boxes.
      boxes2: a numpy array with shape [M, 4] holding N boxes.
    Returns:
      a numpy array with shape [N, M] representing pairwise ioa scores.
    """
    intersect = intersection(boxes1, boxes2)
    inv_areas = np.expand_dims(1.0 / area(boxes2), axis=0)
    return intersect * inv_areas


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
    rpn_proposals[rpn_proposals < min_value] = min_value
    rpn_proposals[:, ::2][rpn_proposals[:, ::2] > max_height - 1.0] = max_height - 1.0
    rpn_proposals[:, 1::2][rpn_proposals[:, 1::2] > max_width - 1.0] = max_width - 1.0

    if min_edge is None:
        return rpn_proposals, np.arange(len(rpn_proposals))

    new_rpn_proposals = []
    rpn_proposals_idx = []
    for idx, (ymin, xmin, ymax, xmax) in enumerate(rpn_proposals):
        if (ymax - ymin + 1.0) >= min_edge and (xmax - xmin + 1.0) >= min_edge:
            new_rpn_proposals.append([ymin, xmin, ymax, xmax])
            rpn_proposals_idx.append(idx)
    return np.array(new_rpn_proposals), np.array(rpn_proposals_idx)


def bboxes_range_filter(anchors, max_height, max_width):
    """
    过滤 anchors，超出图像范围的 anchors 都不要
    :param anchors:
    :param max_height:
    :param max_width:
    :return:
    """
    index_inside = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] <= max_height - 1) &
        (anchors[:, 3] <= max_width - 1)
    )[0]
    return index_inside
