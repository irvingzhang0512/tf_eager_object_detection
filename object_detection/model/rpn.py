import tensorflow as tf
import numpy as np
from object_detection.utils.bbox_transform import decode_bbox
from object_detection.utils.bbox_tf import pairwise_iou

layers = tf.keras.layers


class RPNHead(tf.keras.Model):
    def __init__(self, num_anchors, weight_decay=0.0005):
        """
        :param num_anchors:
        """
        super().__init__()
        self._rpn_conv = layers.Conv2D(512, [3, 3],
                                       padding='same', name='rpn_first_conv',
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),)
        self._rpn_bn = layers.BatchNormalization()

        self._rpn_score_conv = layers.Conv2D(num_anchors * 2, [1, 1],
                                             padding='valid', name='rpn_score_conv',
                                             kernel_initializer='he_normal',
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))

        self._rpn_bbox_conv = layers.Conv2D(num_anchors * 4, [1, 1],
                                            padding='valid', name='rpn_bbox_conv',
                                            kernel_initializer='he_normal',
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
        x = self._rpn_bn(x, training)
        x = tf.nn.relu(x)

        rpn_score = self._rpn_score_conv(x)
        rpn_score_reshape = tf.reshape(rpn_score, [-1, 2])
        rpn_score_softmax = tf.nn.softmax(rpn_score_reshape)

        rpn_bbox = self._rpn_bbox_conv(x)
        rpn_bbox_reshape = tf.reshape(rpn_bbox, [-1, 4])

        return rpn_score_softmax, rpn_bbox_reshape


class RPNTrainingProposal(tf.keras.Model):
    def __init__(self,
                 pos_iou_threshold=0.7,
                 neg_iou_threshold=0.3,
                 total_num_samples=256,
                 max_pos_samples=128, ):
        super().__init__()

        self._pos_iou_threshold = pos_iou_threshold
        self._neg_iou_threshold = neg_iou_threshold
        self._total_num_samples = total_num_samples
        self._max_pos_samples = max_pos_samples

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
        6. 最终输出三个结果：
                1）参与训练的 anchor 在原始anchors中的编号（指在对anchors过滤前，anchors的编号）
                2）每个参与训练的anchor的label（正例还是反例）
                3）每个参与训练的anchor对应的gt_bboxes编号（即与参与训练anchor的iou最大的gt_bboxes编号）
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        anchors, gt_bboxes, image_shape = inputs

        # 1. 对 anchors 进行过滤，筛选符合边界要求的 anchor，之后操作都基于筛选后的结果。
        selected_anchor_idx = _anchors_filter(anchors, image_shape[0], image_shape[1])
        anchors = anchors[selected_anchor_idx]
        # print('after filter has %d anchors' % anchors.shape[0])

        # 2. 计算 anchors 与gt_bboxes（即输入数据中的bbox）的iou。
        labels = -tf.ones((anchors.shape[0],), tf.int32)
        iou = pairwise_iou(anchors, gt_bboxes)  # [anchors_size, gt_bboxes_size]

        # 3. 设置与 gt_bboxes 的 max_iou > pos_iou_threshold 的anchor为正例
        #    设置 max_iou < neg_iou_threshold 的anchor为反例。
        max_ious = tf.reduce_max(iou, axis=1)
        argmax_ious = tf.argmax(iou, axis=1)
        labels = tf.where(max_ious > self._pos_iou_threshold, tf.ones_like(labels), labels)
        labels = tf.where(max_ious < self._neg_iou_threshold, tf.zeros_like(labels), labels)

        # # TODO: 使用 tensorflow 实现下面 numpy 操作
        # # TODO: 存在bug，假设：所有gts中与anchor1 iou最大的是gt1，所有anchors中与gt2 iou最大的是anchor1
        # # TODO: 此时 argmax_ious 可能存在问题
        # # 4. 设置与每个 gt_bboxes 的iou最大的anchor为正例。
        # # 获取与每个 gt_bboxes 的 iou 最大的anchor的编号，设置这些anchor为正例
        # # labels[gt_argmax_ious] = 1
        # gt_argmax_ious = tf.argmax(iou, axis=0)  # [gt_bboxes_size]
        # cond = np.zeros([anchors.shape[0]], dtype=np.int32)
        # cond[gt_argmax_ious.numpy()] = 1
        # labels = tf.where(tf.equal(cond, 1), tf.ones_like(labels), labels)

        # 计算正反例真实数量
        total_pos_num = tf.reduce_sum(tf.where(tf.equal(labels, 1), tf.ones_like(labels), tf.zeros_like(labels)))
        total_neg_num = tf.reduce_sum(tf.where(tf.equal(labels, 0), tf.ones_like(labels), tf.zeros_like(labels)))

        # 根据要求，修正正反例数量
        cur_pos_num = tf.minimum(total_pos_num, self._max_pos_samples)
        cur_neg_num = tf.minimum(self._total_num_samples - cur_pos_num, total_neg_num)
        # print('rpn training has %d pos samples and %d neg samples' % (cur_pos_num, cur_neg_num))

        # 随机选择正例和反例
        total_pos_index = tf.squeeze(tf.where(tf.equal(labels, 1)), axis=1)
        total_neg_index = tf.squeeze(tf.where(tf.equal(labels, 0)), axis=1)
        pos_index = tf.gather(total_pos_index, tf.random_shuffle(tf.range(0, total_pos_num))[:cur_pos_num])
        neg_index = tf.gather(total_neg_index, tf.random_shuffle(tf.range(0, total_neg_num))[:cur_neg_num])

        # 该编号是 anchors filter 之后的编号，而不是原始anchors中的编号
        selected_idx = tf.concat([pos_index, neg_index], axis=0)

        # 生成最终结果
        return tf.stop_gradient(tf.gather(selected_anchor_idx, selected_idx)), \
               tf.stop_gradient(tf.gather(labels, selected_idx)), \
               tf.stop_gradient(tf.gather(argmax_ious, selected_idx)), \
               tf.stop_gradient(cur_pos_num)


class RPNProposal(tf.keras.Model):
    def __init__(self,
                 num_pre_nms_train=12000,
                 num_post_nms_train=2000,
                 num_pre_nms_test=6000,
                 num_post_nms_test=300,
                 nms_iou_threshold=0.7, ):
        super().__init__()

        self._num_pre_nms_train = num_pre_nms_train
        self._num_post_nms_train = num_post_nms_train
        self._num_pre_nms_test = num_pre_nms_test
        self._num_post_nms_test = num_post_nms_test
        self._nms_iou_threshold = nms_iou_threshold

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
        # print(np.max(bboxes_txtytwth), np.min(bboxes_txtytwth))
        decoded_bboxes = decode_bbox(anchors, bboxes_txtytwth)

        # 2. 对选中修正后的anchors进行处理
        decoded_bboxes, selected_idx = proposal_filter(decoded_bboxes,
                                                        0, image_shape[0], image_shape[1], extractor_stride)
        scores = tf.gather(scores, selected_idx)

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

        # print('rpn proposal net generate %d proposals' % tf.size(selected_idx))

        return tf.gather(decoded_bboxes, selected_idx)


def proposal_filter(rpn_proposals, min_value, max_height, max_width, min_edge=None):
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
    if isinstance(rpn_proposals, tf.Tensor):
        rpn_proposals = rpn_proposals.numpy()

    rpn_proposals[rpn_proposals < min_value] = min_value
    rpn_proposals[:, ::2][rpn_proposals[:, ::2] > max_height] = max_height
    rpn_proposals[:, 1::2][rpn_proposals[:, 1::2] > max_width] = max_width

    if min_edge is None:
        return rpn_proposals, np.arange(len(rpn_proposals))

    new_rpn_proposals = []
    rpn_proposals_idx = []
    for idx, (ymin, xmin, ymax, xmax) in enumerate(rpn_proposals):
        if (ymax - ymin) >= min_edge and (xmax - xmin) >= min_edge:
            new_rpn_proposals.append([ymin, xmin, ymax, xmax])
            rpn_proposals_idx.append(idx)
    return np.array(new_rpn_proposals), np.array(rpn_proposals_idx)


def _anchors_filter(anchors, max_height, max_width):
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
        (anchors[:, 2] <= max_height) &
        (anchors[:, 3] <= max_width)
    )[0]
    return index_inside
