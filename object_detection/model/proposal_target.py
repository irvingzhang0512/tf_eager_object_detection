import tensorflow as tf
import numpy as np

from object_detection.utils.bbox_tf import pairwise_iou
from object_detection.utils.bbox_transform import encode_bbox_with_mean_and_std


class ProposalTarget(tf.keras.Model):
    def __init__(self,
                 num_classes=21,
                 pos_iou_threshold=0.5,
                 neg_iou_threshold=0.5,
                 total_num_samples=128,
                 max_pos_samples=32,
                 target_means=None,
                 target_stds=None):
        super().__init__()

        self._num_classes = num_classes
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
        生成训练roi用的数据
        总体过程：
        1. 计算 rois 与 gt_bboxes（即输入数据中的bbox）的iou
        2. 设置与 gt_bboxes 的 max_iou > pos_iou_threshold 的 roi 为正例，设置 max_iou < neg_iou_threshold 的 roi 为反例
        3. 对正例、反例有数量限制：
            正例数量不大于 max_pos_samples
            正例反例总数不超过 max_pos_samples
            反例数量如果过少，则通过 numpy.random.choice 随机填充
        4. 最终输出5个结果：
                1）rois [128, 4]
                2）每个 roi 对应的 label [128,]，如果我为0则表示为反例，>0则表示为正例
                3）每个 roi 对应的 txtytwth [128, num_classes * 4]
                4）计算 smooth l1 loss时的 bbox_inside_weights [128, num_classes * 4]
                5）计算 smooth l1 loss时的 bbox_outside_weights [128, num_classes * 4]
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        rois, gt_bboxes, gt_labels = inputs

        iou = pairwise_iou(rois, gt_bboxes)  # [rois_size, gt_bboxes_size]
        max_overlaps = tf.reduce_max(iou, axis=1)  # [rois_size, ]
        gt_assignment = tf.argmax(iou, axis=1)  # [rois_size, ]
        labels = tf.gather(gt_labels, gt_assignment)  # [rois_size, ]

        # 根据条件获取 前景 背景
        fg_inds = tf.where(max_overlaps >= self._pos_iou_threshold)[:, 0]
        bg_inds = tf.where(tf.logical_and(max_overlaps < self._pos_iou_threshold,
                                          max_overlaps >= self._neg_iou_threshold))[:, 0]

        # 筛选 前景/背景
        if tf.size(fg_inds) > self._max_pos_samples:
            fg_inds = tf.random_shuffle(fg_inds)[:self._max_pos_samples]
        if tf.size(bg_inds) > self._total_num_samples - tf.size(fg_inds):
            # 如果bg sample的数量多于要求值，则随机筛选
            bg_inds = tf.random_shuffle(bg_inds)[:(self._total_num_samples - tf.size(fg_inds))]
        elif tf.size(bg_inds).numpy() == (self._total_num_samples - tf.size(fg_inds)).numpy():
            pass
        else:
            # 如果bg sample的数量少于要求数值，则重复获取
            target_size = (self._total_num_samples - tf.size(fg_inds)).numpy()
            bg_inds = np.random.choice(bg_inds.numpy(), size=int(target_size), replace=True)

        tf.logging.debug('proposal target generate %d fgs and %d bgs.' % (tf.size(fg_inds), tf.size(bg_inds)))

        keep_inds = tf.concat([fg_inds, bg_inds], axis=0)
        final_rois = tf.gather(rois, keep_inds)  # rois[keep_inds]
        final_labels = tf.gather(labels, keep_inds)  # labels[keep_inds]
        # labels[fg_inds_size:] = 0
        final_labels = tf.scatter_update(tf.Variable(final_labels),
                                         tf.range(tf.size(fg_inds), tf.size(keep_inds), dtype=tf.int32), 0)

        # inside weights 只有正例才会设置，其他均为0
        bbox_inside_weights = tf.zeros((tf.size(keep_inds), self._num_classes, 4), dtype=tf.float32)
        if tf.size(fg_inds) > 0:
            # memory leak bug for tf.scatter_nd_update
            # https://github.com/tensorflow/tensorflow/issues/27288
            # cur_index = tf.stack([tf.range(tf.size(fg_inds)), tf.gather(labels, fg_inds)], axis=1)
            # bbox_inside_weights = tf.scatter_nd_update(tf.Variable(bbox_inside_weights),
            #                                            cur_index,
            #                                            tf.ones([tf.size(fg_inds), 4]))
            bbox_inside_weights = bbox_inside_weights.numpy()
            for idx, fg_ind in enumerate(fg_inds.numpy()):
                bbox_inside_weights[idx, labels[idx]] = 1
        bbox_inside_weights = tf.reshape(bbox_inside_weights, [-1, self._num_classes * 4])

        # final bbox target 只有正例才会设置，其他均为0
        final_bbox_targets = tf.zeros((tf.size(keep_inds), self._num_classes, 4), dtype=tf.float32)
        if tf.size(fg_inds) > 0:
            bbox_targets = encode_bbox_with_mean_and_std(tf.gather(final_rois, tf.range(tf.size(fg_inds))),
                                                         tf.gather(gt_bboxes, tf.gather(gt_assignment, fg_inds)),
                                                         target_stds=self._target_stds, target_means=self._target_means,
                                                         )
            # memory leak bug for tf.scatter_nd_update
            # https://github.com/tensorflow/tensorflow/issues/27288
            # final_bbox_targets = tf.scatter_nd_update(tf.Variable(final_bbox_targets),
            #                                           tf.stack([tf.range(tf.size(fg_inds)),
            #                                                     tf.gather(labels, fg_inds)], axis=1), bbox_targets)
            final_bbox_targets = final_bbox_targets.numpy()
            bbox_targets = bbox_targets.numpy()
            for idx, fg_ind in enumerate(fg_inds.numpy()):
                final_bbox_targets[idx, labels[idx]] = bbox_targets[idx]

        final_bbox_targets = tf.reshape(final_bbox_targets, [-1, self._num_classes * 4])

        # 这个好像没啥用
        bbox_outside_weights = tf.ones_like(bbox_inside_weights, dtype=tf.float32)
        return tf.stop_gradient(final_rois), tf.stop_gradient(final_labels), tf.stop_gradient(final_bbox_targets), \
               tf.stop_gradient(bbox_inside_weights), tf.stop_gradient(bbox_outside_weights)
