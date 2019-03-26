import numpy as np
import tensorflow as tf
from six.moves import range

__all__ = ['generate_anchor_base', 'generate_by_anchor_base_np', 'generate_by_anchor_base_tf', 'make_anchors']

"""
参考了多处代码，包括：
Numpy实现主要参考了：
https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/generate_anchors.py

TF实现主要参考了（其实主要还是参考了Numpy代码，只是改写成TF）：
https://github.com/Viredery/tf-eager-fasterrcnn/blob/master/detection/core/anchor/anchor_generator.py


使用方式：
要么使用 `generate_anchor_base` 和 `generate_by_anchor_base_np`/`generate_by_anchor_base_tf`
要么使用 `generate_anchors_tf` 或 `generate_anchors_np`
具体实例参考本文件主函数中内容
"""


def generate_by_anchor_base_np(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    import numpy as xp
    shift_y = xp.arange(0, height, feat_stride)
    shift_x = xp.arange(0, width, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def generate_by_anchor_base_tf(anchor_base, feat_stride, height, width):
    shift_x = tf.range(width) * feat_stride  # width
    shift_y = tf.range(height) * feat_stride  # height
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    sx = tf.reshape(shift_x, shape=(-1,))
    sy = tf.reshape(shift_y, shape=(-1,))
    shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))

    K = tf.multiply(width, height)
    A = anchor_base.shape[0]
    shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]), perm=(1, 0, 2))
    anchor_constant = tf.to_float(tf.reshape(anchor_base, (1, A, 4)))
    anchors_tf = tf.reshape(tf.add(anchor_constant, tf.to_float(shifts)), shape=(-1, 4))

    return tf.cast(anchors_tf, dtype=tf.float32)


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         scales=2 ** np.arange(3, 6)):
    """
    有两种生成 anchor base 的方法，这种好像是原论文中使用的
    anchor base 决定了最终 anchors 的长宽，后续 generate_by_anchor_base 函数的作用是确定anchor的中心点
    输入的三个参数都会影响到最终的长宽：
    ratios 确定了长宽的比例
    base_size 和 scales 共同决定了 anchor 的具体尺寸，即 base_size * scales 就是最终 anchors 的尺寸
      Generate anchor (reference) windows by enumerating aspect ratios X
      scales wrt a reference (0, 0, 15, 15) window.
    """

    ratios = np.array(ratios)
    scales = np.array(scales)
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors


def _whctrs(anchor):
    """
  Return width, height, x center, and y center for an anchor (window).
  """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
  Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).
  """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
  Enumerate a set of anchors for each aspect ratio wrt an anchor.
  """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
  Enumerate a set of anchors for each scale wrt an anchor.
  """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def make_anchors(base_anchor_size, anchor_scales, anchor_ratios,
                 featuremap_height, featuremap_width,
                 stride, name='make_anchors'):
    with tf.variable_scope(name):
        base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], tf.float32)  # [x_center, y_center, w, h]

        ws, hs = enum_ratios(enum_scales(base_anchor, anchor_scales),
                             anchor_ratios)  # per locations ws and hs

        x_centers = tf.range(featuremap_width, dtype=tf.float32) * stride
        y_centers = tf.range(featuremap_height, dtype=tf.float32) * stride

        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

        ws, x_centers = tf.meshgrid(ws, x_centers)
        hs, y_centers = tf.meshgrid(hs, y_centers)

        anchor_centers = tf.stack([x_centers, y_centers], 2)
        anchor_centers = tf.reshape(anchor_centers, [-1, 2])

        box_sizes = tf.stack([ws, hs], axis=2)
        box_sizes = tf.reshape(box_sizes, [-1, 2])
        # anchors = tf.concat([anchor_centers, box_sizes], axis=1)
        anchors = tf.concat([anchor_centers - 0.5 * box_sizes,
                             anchor_centers + 0.5 * box_sizes], axis=1)
        return anchors


def enum_scales(base_anchor, anchor_scales):
    anchor_scales = base_anchor * tf.constant(anchor_scales, dtype=tf.float32, shape=(len(anchor_scales), 1))
    return anchor_scales


def enum_ratios(anchors, anchor_ratios):
    ws = anchors[:, 2]  # for base anchor: w == h
    hs = anchors[:, 3]
    sqrt_ratios = tf.sqrt(tf.constant(anchor_ratios))

    ws = tf.reshape(ws / sqrt_ratios[:, tf.newaxis], [-1, 1])
    hs = tf.reshape(hs * sqrt_ratios[:, tf.newaxis], [-1, 1])

    return hs, ws
