import numpy as np


def encode_bbox(src_bbox, dst_bbox):
    """
    https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/model/utils/bbox_tools.py
    :param src_bbox: (N, 4), ymin xmin ymax xmax
    :param dst_bbox: (N, 4), ymin xmin ymax xmax
    :return:
    """
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
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


def decode_bbox(anchors, bboxes_txtytwth):
    """
    https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/model/utils/bbox_tools.py
    :param anchors:  (N, 4), ymin xmin ymax xmax
    :param bboxes_txtytwth:  (N, 4), dy dx dh dw
    :return:
    """
    src_bbox = anchors.astype(anchors.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
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

    dst_bbox = np.zeros(src_height.shape, dtype=src_height.dtype)
    dst_bbox[:, 0] = ctr_y - 0.5 * h
    dst_bbox[:, 1] = ctr_x - 0.5 * w
    dst_bbox[:, 2] = ctr_y + 0.5 * h
    dst_bbox[:, 3] = ctr_x + 0.5 * w
    return bboxes
