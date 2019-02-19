import tensorflow as tf
from object_detection.utils.bbox_transform import decode_bbox_with_mean_and_std, encode_bbox_with_mean_and_std
from object_detection.utils.bbox_tf import pairwise_iou, bboxes_clip_filter, bboxes_range_filter
from tensorflow.python.platform import tf_logging

layers = tf.keras.layers

__all__ = ['RPNHead', 'RPNTrainingProposal', 'RPNProposal']


class RPNHead(tf.keras.Model):
    def __init__(self, num_anchors, weight_decay=0.0005):
        """
        :param num_anchors:
        """
        super().__init__()
        self._rpn_conv = layers.Conv2D(512, [3, 3],
                                       padding='same', name='rpn_first_conv', activation='relu',
                                       kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay), )

        self._rpn_score_conv = layers.Conv2D(num_anchors * 2, [1, 1],
                                             padding='valid', name='rpn_score_conv',
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))

        self._rpn_bbox_conv = layers.Conv2D(num_anchors * 4, [1, 1],
                                            padding='valid', name='rpn_bbox_conv',
                                            activation=None,
                                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay))

    def call(self, inputs, training=None, mask=None):
        """
        参与训练，不能使用numpy操作
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        x = self._rpn_conv(inputs)

        rpn_score = self._rpn_score_conv(x)
        rpn_score_reshape = tf.reshape(rpn_score, [-1, 2])

        rpn_bbox = self._rpn_bbox_conv(x)
        rpn_bbox_reshape = tf.reshape(rpn_bbox, [-1, 4])

        return rpn_score_reshape, rpn_bbox_reshape


class RPNTrainingProposal(tf.keras.Model):
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
        anchors, gt_bboxes, image_shape = inputs

        # 1. 对 anchors 进行过滤，筛选符合边界要求的 anchor，之后操作都基于筛选后的结果。
        # # np 实现
        # selected_anchor_idx = _anchors_filter(anchors, image_shape[0], image_shape[1])
        # anchors = anchors[selected_anchor_idx]
        # tf实现
        tf_logging.debug('rpn training, before filter has %d anchors' % anchors.shape[0])
        selected_anchor_idx = bboxes_range_filter(anchors, image_shape[0], image_shape[1])
        anchors = tf.gather(anchors, selected_anchor_idx)
        tf_logging.debug('rpn training, after filter has %d anchors' % anchors.shape[0])

        # 2. 计算 anchors 与gt_bboxes（即输入数据中的bbox）的iou。
        labels = -tf.ones((anchors.shape[0],), tf.int32)
        iou = pairwise_iou(anchors, gt_bboxes)  # [anchors_size, gt_bboxes_size]

        # 3. 设置与 gt_bboxes 的 max_iou > pos_iou_threshold 的anchor为正例
        #    设置 max_iou < neg_iou_threshold 的anchor为反例。
        max_ious = tf.reduce_max(iou, axis=1)
        argmax_ious = tf.argmax(iou, axis=1, output_type=tf.int32)
        labels = tf.where(max_ious >= self._pos_iou_threshold, tf.ones_like(labels), labels)
        labels = tf.where(max_ious < self._neg_iou_threshold, tf.zeros_like(labels), labels)

        # 4. 设置与每个 gt_bboxes 的iou最大的anchor为正例。
        # 获取与每个 gt_bboxes 的 iou 最大的anchor的编号，设置这些anchor为正例，并修改对应 argmax_ious
        # 想要实现 labels[gt_argmax_ious] = 1, argmax_ious[gt_argmax_ious] = np.arange(len(gt_argmax_ious))
        gt_argmax_ious = tf.argmax(iou, axis=0, output_type=tf.int32)  # [gt_bboxes_size]
        labels = tf.scatter_update(tf.Variable(labels), gt_argmax_ious, 1)
        argmax_ious = tf.scatter_update(tf.Variable(argmax_ious), gt_argmax_ious, tf.range(0, tf.size(gt_argmax_ious)))

        # 筛选正例和反例
        pos_index = tf.where(tf.equal(labels, 1))[:, 0]
        neg_index = tf.where(tf.equal(labels, 0))[:, 0]
        total_pos_num = tf.size(pos_index)  # 计算正例真实数量
        total_neg_num = tf.size(neg_index)  # 计算正例真实数量
        cur_pos_num = tf.minimum(total_pos_num, self._max_pos_samples)  # 根据要求，修正正例数量
        cur_neg_num = tf.minimum(self._total_num_samples - cur_pos_num, total_neg_num)  # 根据要求，修正反例数量
        pos_index = tf.random_shuffle(pos_index)[:cur_pos_num]  # 随机选择正例
        neg_index = tf.random_shuffle(neg_index)[:cur_neg_num]  # 随机选择反例
        tf_logging.debug('rpn training has %d pos samples and %d neg samples' % (cur_pos_num, cur_neg_num))

        # 该编号是 anchors filter 之后的编号，而不是原始anchors中的编号
        selected_idx = tf.concat([pos_index, neg_index], axis=0)

        # 计算 rpn training 中 reg loss 的 gt
        selected_gt_bboxes = tf.gather(gt_bboxes, tf.gather(argmax_ious, pos_index))
        selected_pred_bboxes = tf.gather(anchors, pos_index)
        rpn_gt_txtytwth = encode_bbox_with_mean_and_std(selected_pred_bboxes, selected_gt_bboxes,
                                                        self._target_means, self._target_stds)

        target_anchors_idx = tf.gather(selected_anchor_idx, selected_idx)
        target_labels = tf.gather(labels, selected_idx)

        # 生成最终结果
        return tf.stop_gradient(target_anchors_idx), tf.stop_gradient(target_labels), \
               tf.stop_gradient(rpn_gt_txtytwth), tf.stop_gradient(cur_pos_num)


class RPNProposal(tf.keras.Model):
    def __init__(self,
                 num_pre_nms_train=12000,
                 num_post_nms_train=2000,
                 num_pre_nms_test=6000,
                 num_post_nms_test=300,
                 nms_iou_threshold=0.7,
                 target_means=None,
                 target_stds=None):
        super().__init__()

        self._num_pre_nms_train = num_pre_nms_train
        self._num_post_nms_train = num_post_nms_train
        self._num_pre_nms_test = num_pre_nms_test
        self._num_post_nms_test = num_post_nms_test
        self._nms_iou_threshold = nms_iou_threshold

        if target_stds is None:
            target_stds = [1, 1, 1, 1]
        if target_means is None:
            target_means = [0, 0, 0, 0]
        self._target_means = target_means
        self._target_stds = target_stds

    def call(self, inputs, training=None, mask=None):
        """
        不参与训练
        生成 rpn 的结果，即一组 bboxes，用于后续 roi pooling
        总体过程：
        1. 使用anchors使用rpn_pred修正，获取所有预测结果。
        2. 对选中修正后的anchors进行处理。
        3. 根据rpn_score获取num_pre_nms个anchors。
        4. 进行nms。
        5. 根据rpn_score排序，获取num_post_nms个anchors作为proposal结果。
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        # [num_anchors*feature_width*feature_height, 4]
        # [num_anchors*feature_width*feature_height, 4]
        # [num_anchors*feature_width*feature_height,]
        # [2, ]
        bboxes_txtytwth, anchors, scores, image_shape, extractor_stride = inputs

        # 1. 使用anchors使用rpn_pred修正，获取所有预测结果。
        # [num_anchors*feature_width*feature_height, 4]
        tf_logging.debug(('rpn head txtytwth max & min',
                          tf.reduce_max(bboxes_txtytwth),
                          tf.reduce_min(bboxes_txtytwth)))
        decoded_bboxes = decode_bbox_with_mean_and_std(anchors, bboxes_txtytwth,
                                                       self._target_means, self._target_stds)

        # 2. 对选中修正后的anchors进行处理
        decoded_bboxes, selected_idx = bboxes_clip_filter(decoded_bboxes,
                                                          0, image_shape[0], image_shape[1], extractor_stride)
        # decoded_bboxes, selected_idx = proposal_filter(decoded_bboxes,
        #                                                0, image_shape[0], image_shape[1], extractor_stride)
        scores = tf.gather(scores, selected_idx)
        tf_logging.debug('rpn after filter has %d proposals' % tf.size(selected_idx))

        # 3. 根据rpn_score获取num_pre_nms个anchors。
        num_pre_nms = self._num_pre_nms_train if training else self._num_pre_nms_test
        cur_top_k = tf.minimum(num_pre_nms, tf.size(scores))
        _, selected_idx = tf.nn.top_k(scores, k=cur_top_k, sorted=False)
        decoded_bboxes = tf.gather(decoded_bboxes, selected_idx)
        scores = tf.gather(scores, selected_idx)

        # 4. 进行nms。
        # 5. 根据rpn_score排序，获取num_post_nms个anchors作为proposal结果。
        num_post_nms = self._num_post_nms_train if training else self._num_post_nms_test
        cur_top_k = tf.minimum(num_post_nms, tf.size(scores))
        selected_idx = tf.image.non_max_suppression(tf.to_float(decoded_bboxes), scores, cur_top_k,
                                                    iou_threshold=self._nms_iou_threshold)

        tf_logging.debug('rpn proposal net generate %d proposals' % tf.size(selected_idx))

        return tf.stop_gradient(tf.gather(decoded_bboxes, selected_idx))
