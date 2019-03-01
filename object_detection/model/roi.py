import tensorflow as tf
from object_detection.utils.bbox_tf import pairwise_iou
from tensorflow.python.platform import tf_logging
from object_detection.utils.bbox_transform import encode_bbox_with_mean_and_std

layers = tf.keras.layers

VGG_16_WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels.h5')

__all__ = ['RoiHead', 'RoiPooling', 'ProposalTarget']


class RoiPooling(tf.keras.Model):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size
        self._concat_layer = layers.Concatenate(axis=0)
        self._flatten_layer = layers.Flatten()
        self._max_pool = layers.MaxPooling2D(padding='same')

    def call(self, inputs, training=None, mask=None):
        """
        输入 backbone 的结果和 rpn proposals 的结果(即 RegionProosal 的输出)
        输出 roi pooloing 的结果，即在特征图上，对每个rpn proposal获取一个固定尺寸的特征图
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        # [1, height, width, channels]  [num_rois, 4]
        shared_layers, rois, extractor_stride = inputs
        rois = rois / extractor_stride

        batch_ids = tf.zeros([tf.shape(rois)[0]], dtype=tf.int32)
        h, w = shared_layers.get_shape().as_list()[1:3]
        roi_channels = tf.split(rois, 4, axis=1)
        bboxes = tf.concat([
            roi_channels[1] / tf.to_float(h - 1),
            roi_channels[0] / tf.to_float(w - 1),
            roi_channels[3] / tf.to_float(h - 1),
            roi_channels[2] / tf.to_float(w - 1),
        ], axis=1)
        pre_pool_size = self._pool_size * 2
        crops = tf.image.crop_and_resize(shared_layers, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                         name="crops")
        return tf.stop_gradient(self._flatten_layer(self._max_pool(crops)))


class RoiHead(tf.keras.Model):
    def __init__(self, num_classes,
                 roi_feature_size=7 * 7 * 512,
                 keep_rate=0.5, weight_decay=0.0005):
        super().__init__()
        self._num_classes = num_classes

        self._fc1 = layers.Dense(4096, name='fc1', activation='relu',
                                 kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                 input_shape=[roi_feature_size]
                                 )
        self._dropout1 = layers.Dropout(rate=1 - keep_rate)

        self._fc2 = layers.Dense(4096, name='fc2', activation='relu',
                                 kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                 )
        self._dropout2 = layers.Dropout(rate=1 - keep_rate)

        self._score_layer = layers.Dense(num_classes, name='roi_head_score', activation=None,
                                         kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                         kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self._roi_bboxes_layer = layers.Dense(4 * num_classes, name='roi_head_bboxes', activation=None,
                                              kernel_initializer=tf.random_normal_initializer(0, 0.001),
                                              kernel_regularizer=tf.keras.regularizers.l2(weight_decay))

        self.build((None, roi_feature_size))
        weights_path = tf.keras.utils.get_file(
            'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
            VGG_16_WEIGHTS_PATH,
            cache_subdir='models',
            file_hash='64373286793e3c8b2b4e3219cbf3544b')
        self.load_weights(weights_path, by_name=True)

    def call(self, inputs, training=None):
        """
        输入 roi pooling 的结果
        对每个 roi pooling 的结果进行预测（预测bboxes）
        :param inputs:  roi_features, [num_rois, len_roi_feature]
        :param training:
        :param mask:
        :return:
        """
        x = self._fc1(inputs)
        x = self._dropout1(x, training)
        x = self._fc2(x)
        x = self._dropout2(x, training)
        score = self._score_layer(x)
        bboxes = self._roi_bboxes_layer(x)

        return score, bboxes


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
        3. 对正例、反例有数量限制，正例数量不大于 max_pos_samples，正例反例总数不超过 max_pos_samples
        4. 最终输出三个结果：
                1）参与训练的 roi 的编号
                2）每个参与训练的 roi 的label [0, num_classes)，可直接用于 cls loss
                3）pos rois 对应的 gt，可直接用于 reg loss
                4）pos anchors num，scalar
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
        fg_inds = tf.where(max_overlaps > self._pos_iou_threshold)[:, 0]
        # bg_inds = tf.where(tf.logical_and(max_overlaps < self._pos_iou_threshold,
        #                                   max_overlaps >= self._neg_iou_threshold))[:, 0]
        bg_inds = tf.where(max_overlaps < self._pos_iou_threshold)[:, 0]

        # 筛选 前景/背景
        if tf.size(fg_inds) > self._max_pos_samples:
            fg_inds = tf.random_shuffle(fg_inds)[:self._max_pos_samples]
        if tf.size(bg_inds) > self._total_num_samples - tf.size(fg_inds):
            bg_inds = tf.random_shuffle(bg_inds)[:(self._total_num_samples - tf.size(fg_inds))]

        keep_inds = tf.concat([fg_inds, bg_inds], axis=0)
        final_rois = tf.gather(rois, keep_inds)  # rois[keep_inds]
        final_labels = tf.gather(labels, keep_inds)  # labels[keep_inds]
        # labels[fg_inds_size:] = 0
        final_labels = tf.scatter_update(tf.Variable(final_labels), tf.range(tf.size(fg_inds), tf.size(keep_inds),
                                                                             dtype=tf.int32), 0)

        # inside weights 只有正例才会设置，其他均为0
        bbox_inside_weights = tf.zeros((tf.size(keep_inds), self._num_classes, 4), dtype=tf.float32)
        if tf.size(fg_inds) > 0:
            cur_index = tf.stack([tf.range(tf.size(fg_inds)), tf.gather(labels, fg_inds)],  axis=1)
            bbox_inside_weights = tf.scatter_nd_update(tf.Variable(bbox_inside_weights),
                                                       cur_index,
                                                       tf.ones([tf.size(fg_inds), 4]))
        bbox_inside_weights = tf.reshape(bbox_inside_weights, [-1, self._num_classes * 4])

        # final bbox target 只有正例才会设置，其他均为0
        final_bbox_targets = tf.zeros((tf.size(keep_inds), self._num_classes, 4), dtype=tf.float32)
        if tf.size(fg_inds) > 0:
            bbox_targets = encode_bbox_with_mean_and_std(tf.gather(final_rois, tf.range(tf.size(fg_inds))),
                                                         tf.gather(gt_bboxes, tf.gather(gt_assignment, fg_inds)),
                                                         target_stds=self._target_stds, target_means=self._target_means,
                                                         )
            final_bbox_targets = tf.scatter_nd_update(tf.Variable(final_bbox_targets),
                                                      tf.stack([tf.range(tf.size(fg_inds)), tf.gather(labels, fg_inds)],
                                                               axis=1), bbox_targets)
        final_bbox_targets = tf.reshape(final_bbox_targets, [-1, self._num_classes * 4])

        # 这个好像没啥用
        bbox_outside_weights = tf.ones_like(bbox_inside_weights, dtype=tf.float32)
        return tf.stop_gradient(final_rois), tf.stop_gradient(final_labels), tf.stop_gradient(final_bbox_targets), \
               tf.stop_gradient(bbox_inside_weights), tf.stop_gradient(bbox_outside_weights)
