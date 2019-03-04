import numpy as np
import tensorflow as tf
from six.moves import range

__all__ = ['generate_anchor_base', 'generate_by_anchor_base_np', 'generate_by_anchor_base_tf',
           'generate_anchors_tf', 'generate_anchors_np']

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
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    shift_y = tf.range(0, height, feat_stride, dtype=tf.float32)
    shift_x = tf.range(0, width, feat_stride, dtype=tf.float32)
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift = tf.stack([tf.reshape(shift_y, [-1]), tf.reshape(shift_x, [-1]),
                      tf.reshape(shift_y, [-1]), tf.reshape(shift_x, [-1])], axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = tf.to_float(tf.reshape(anchor_base, [1, A, 4])) + tf.transpose(tf.reshape(shift, [1, K, 4]), [1, 0, 2])
    anchor = tf.to_float(tf.reshape(anchor, [K * A, 4]))
    return anchor


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         scales=2 ** np.arange(3, 6)):
    """
    有两种生成 anchor base 的方法，这种好像是原论文中使用的
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


def generate_anchors_np(scales, ratios, shape, feature_stride=1, anchor_stride=1):
    """
    记录一下用法：
    如果shape是feature map的尺寸，若要在生成原始图像尺寸上的anchors，scales 必须是原始图像上的尺寸, 需要设置 feature stride
    如果shape是原始图像的纯，若要生成原始图像尺寸上的anchors，scales 必须是原始图像上的尺寸，需要设置 anchor_stride

    简单说，
    anchors 的长宽的是 scales / sqrt(ratios)
    anchors 的中心点位置的是 range(0, shape[0], anchor_stride) * feature_stride
                             range(0, shape[1], anchor_stride) * feature_stride
    anchors 数量的是 len(scales) * len(ratios) * (shape[0] // anchor_stride) * (shape[1] // anchor_stride)

    copy from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_anchors_tf(ratios, scales, shape, feature_stride=1, anchor_stride=1):
    """Generate the anchors given the spatial shape of feature map.
    记录一下用法：
    如果shape是feature map的尺寸，若要在生成原始图像尺寸上的anchors，scales 必须是原始图像上的尺寸, 需要设置 feature stride
    如果shape是原始图像的纯，若要生成原始图像尺寸上的anchors，scales 必须是原始图像上的尺寸，需要设置 anchor_stride

    简单说，
    anchors 的长宽的是 scales / sqrt(ratios)
    anchors 的中心点位置的是 range(0, shape[0], anchor_stride) * feature_stride
                             range(0, shape[1], anchor_stride) * feature_stride
    anchors 数量的是 len(scales) * len(ratios) * (shape[0] // anchor_stride) * (shape[1] // anchor_stride)
    """
    scales = tf.to_float(scales)

    # Get all combinations of scales and ratios
    scales, ratios = tf.meshgrid(scales, ratios)
    scales = tf.reshape(scales, [-1])
    ratios = tf.reshape(ratios, [-1])

    # Enumerate heights and widths from scales and ratios
    heights = scales / tf.sqrt(ratios)
    widths = scales * tf.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = tf.multiply(tf.range(0, shape[0], anchor_stride), feature_stride)
    shifts_x = tf.multiply(tf.range(0, shape[1], anchor_stride), feature_stride)

    shifts_x, shifts_y = tf.cast(shifts_x, tf.float32), tf.cast(shifts_y, tf.float32)
    shifts_x, shifts_y = tf.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = tf.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = tf.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = tf.reshape(tf.stack([box_centers_y, box_centers_x], axis=2), (-1, 2))
    box_sizes = tf.reshape(tf.stack([box_heights, box_widths], axis=2), (-1, 2))

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = tf.concat([box_centers - 0.5 * box_sizes,
                       box_centers + 0.5 * box_sizes], axis=1)
    return boxes


if __name__ == '__main__':
    image_shape = np.array([600, 800])
    extractor_stride = 16
    shared_feature_shape = image_shape // extractor_stride

    cur_ratios = [0.5, 1.0, 2.0]
    scales_feature_map = np.array([8, 16, 32])
    scales_image = scales_feature_map * extractor_stride

    # 如果要生成原始图像尺寸上的 anchors，有以下两类方法

    # method 1
    # anchors 的边长通过 scales 和 cur_ratios 决定，即 scales / sqrt(ratios)
    # anchors 的中心点坐标通过 shape, anchor_stride, feature_stride 决定，即：
    #                               range(0, shape[0], anchor_stride) * feature_stride
    #                               range(0, shape[1], anchor_stride) * feature_stride
    # anchors 的数量通过 scales, ratios, shape, anchor_stride 决定，即：
    #                           len(scales) * len(ratios) * (shape[0] // anchor_stride) * (shape[1] // anchor_stride)
    # 以下三种方式生成的结果相同
    anchors1 = generate_anchors_np(scales_feature_map, cur_ratios, shared_feature_shape,
                                   feature_stride=1, anchor_stride=1) * extractor_stride
    anchors2 = generate_anchors_np(scales_image, cur_ratios, shared_feature_shape,
                                   feature_stride=extractor_stride, anchor_stride=1)
    anchors3 = generate_anchors_np(scales_image, cur_ratios, image_shape,
                                   feature_stride=1, anchor_stride=extractor_stride)

    # method 2
    # anchors_base 可以理解为其 ymin xmin ymax xmax 以某个点为中心的偏移量
    # 默认实现中，是以 (px, py) 为中心的变化量，py = base_size / 2. px = base_size / 2
    # 我也不知道为啥要用 py px 作为中心
    # anchor_base 决定了 anchors 的边长，其计算方法为 base_size * anchor_scales[j]，所以需要使用的是 feature_map shape
    cur_anchor_base = generate_anchor_base(extractor_stride, cur_ratios, scales_feature_map)

    # generate_by_anchor_base 可以看做是生成 anchor 中心点的过程
    # 中心点生成后，通过 anchor_base 得到的 ymin, xmin, ymax, xmax 根据中心点的偏移量，可以直接获得最终结果anchors
    # 中心点的生成方式是 np.arange(0, height * feat_stride, feat_stride)
    # 所以，输入原始图像的 shape 以及 extractor_stride即可
    # anchors4 = generate_by_anchor_base_np(cur_anchor_base, extractor_stride, image_shape[0], image_shape[1])
    anchors4 = generate_by_anchor_base_tf(cur_anchor_base, extractor_stride, image_shape[0], image_shape[1])
    # anchors4 和 anchors1/2/3 不同，原因就是在 anchor_base 实现中，使用了 py px 作为中心点
