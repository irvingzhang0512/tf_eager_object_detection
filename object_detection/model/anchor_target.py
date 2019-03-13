import tensorflow as tf
from object_detection.utils.bbox_transform import encode_bbox_with_mean_and_std
from object_detection.utils.bbox_tf import pairwise_iou, bboxes_range_filter
from tensorflow.python.platform import tf_logging


class AnchorTarget(tf.keras.Model):
    def __init__(self,
                 pos_iou_threshold=0.7,
                 neg_iou_threshold=0.3,
                 total_num_samples=256,
                 max_pos_samples=128,
                 target_means=None,
                 target_stds=None):
        super().__init__()

        self._pos_iou_threshold = pos_iou_threshold
        self._neg_iou_threshold = neg_iou_threshold
        self._total_num_samples = total_num_samples
        self._max_pos_samples = max_pos_samples

        if target_stds is None:
            target_stds = [1, 1, 1, 1]
        if target_means is None:
            target_means = [0, 0, 0, 0]
        self._target_means = target_means
        self._target_stds = target_stds

    def call(self, inputs, training=None, mask=None):
        """
        不需要训练
        生成训练rpn用的训练数据
        总体过程：
        1. 对 anchors 进行过滤，筛选符合边界要求的 anchor，之后操作都基于筛选后的结果。
        2. 计算 anchors 与gt_bboxes（即输入数据中的bbox）的iou。
        3. 设置与 gt_bboxes 的 max_iou > 0.7的anchor为正例，设置 max_iou < 0.3 的anchor为反例。
        4. 设置与每个 gt_bboxes 的iou最大的anchor为正例。
        5. 对正例、反例有数量限制，正例数量不大于 max_pos_samples，正例反例总数不超过 max_pos_samples。
        6. 最终输出4个结果：
                1）参与训练的 anchor 在原始 anchors 中的编号, [training_anchors_num, ], tf.int32
                2）每个参与训练的anchor的label（正例还是反例，顺序与前一结果对应）, [training_anchors_num, ], tf.int32
                3）rpn training reg loss 中对应的 gt，[training_pos_anchors_num, 4], tf.float32
                4）正例数量，scalar，tf.int32
                PS: 保证正例idx、label在前面，反例idx、label在后面
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        gt_bboxes, image_shape, all_anchors, num_anchors = inputs
        total_anchors = all_anchors.get_shape().as_list()[0]

        # 1. 对 anchors 进行过滤，筛选符合边界要求的 anchor，之后操作都基于筛选后的结果。
        tf_logging.debug('rpn training, before filter has %d anchors' % all_anchors.shape[0])
        selected_anchor_idx = bboxes_range_filter(all_anchors, image_shape[0], image_shape[1])
        anchors = tf.gather(all_anchors, selected_anchor_idx)
        tf_logging.debug('rpn training, after filter has %d anchors' % anchors.shape[0])

        # 准备工作
        labels = -tf.ones((anchors.shape[0],), tf.int32)
        overlaps = pairwise_iou(anchors, gt_bboxes)  # [anchors_size, gt_bboxes_size]
        argmax_overlaps = tf.argmax(overlaps, axis=1, output_type=tf.int32)
        max_overlaps = tf.reduce_max(overlaps, axis=1)
        gt_max_overlaps = tf.reduce_max(overlaps, axis=0)
        gt_argmax_overlaps = tf.where(tf.equal(overlaps, gt_max_overlaps))[:, 0]

        # 设置labels
        labels = tf.where(max_overlaps < self._neg_iou_threshold, tf.zeros_like(labels), labels)
        labels = tf.scatter_update(tf.Variable(labels), gt_argmax_overlaps, 1)
        labels = tf.where(max_overlaps >= self._pos_iou_threshold, tf.ones_like(labels), labels)

        # 筛选正例反例
        fg_inds = tf.where(tf.equal(labels, 1))[:, 0]
        if tf.size(fg_inds) > self._max_pos_samples:
            disable_inds = tf.random_shuffle(fg_inds)[self._max_pos_samples:]
            labels = tf.scatter_update(tf.Variable(labels), disable_inds, -1)
        num_bg = self._total_num_samples - tf.reduce_sum(tf.to_int32(tf.equal(labels, 1)))
        bg_inds = tf.where(tf.equal(labels, 0))[:, 0]
        if tf.size(bg_inds) > num_bg:
            bg_inds = tf.random_shuffle(bg_inds)
            disable_inds = bg_inds[num_bg:]
            bg_inds = bg_inds[:num_bg]
            labels = tf.scatter_update(tf.Variable(labels), disable_inds, -1)
        tf.logging.debug('anchor target generate %d fgs and %d bgs.' % (tf.size(fg_inds), tf.size(bg_inds)))

        # 计算 bboxes targets，作为 rpn reg loss 的 ground truth
        bboxes_targets = encode_bbox_with_mean_and_std(anchors, tf.gather(gt_bboxes, argmax_overlaps),
                                                       target_means=self._target_means,
                                                       target_stds=self._target_stds)

        # 只有整理才有 reg loss
        bbox_inside_weights = tf.zeros((anchors.shape[0], 4), dtype=tf.float32)
        bbox_inside_weights = tf.scatter_update(tf.Variable(bbox_inside_weights),
                                                tf.where(tf.equal(labels, 1))[:, 0], 1)

        # 实质就是对 reg loss / num_rpn_samples
        bbox_outside_weights = tf.zeros((anchors.shape[0], 4), dtype=tf.float32)
        num_examples = tf.reduce_sum(tf.to_float(labels >= 0))
        bbox_outside_weights = tf.scatter_update(tf.Variable(bbox_outside_weights),
                                                 tf.where(labels >= 0)[:, 0], 1.0 / num_examples)

        # 生成最终结果
        return tf.stop_gradient(_unmap(labels, total_anchors, selected_anchor_idx, -1)), \
               tf.stop_gradient(_unmap(bboxes_targets, total_anchors, selected_anchor_idx, 0)), \
               tf.stop_gradient(_unmap(bbox_inside_weights, total_anchors, selected_anchor_idx, 0)), \
               tf.stop_gradient(_unmap(bbox_outside_weights, total_anchors, selected_anchor_idx, 0))


def _unmap(data, count, inds, fill=0):
    """
    将 filter anchors 后的结果映射到 原始 anchors 中，主要就是 index 的转换
    :param data:
    :param count:
    :param inds:
    :param fill:
    :return:
    """
    if len(data.shape) == 1:
        ret = tf.ones([count], dtype=tf.float32) * fill
        ret = tf.scatter_update(tf.Variable(ret), inds, tf.to_float(data))
    else:
        ret = tf.ones([count, ] + data.get_shape().as_list()[1:], dtype=tf.float32) * fill
        ret = tf.scatter_update(tf.Variable(ret), inds, tf.to_float(data))
    return ret
