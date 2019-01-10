import tensorflow as tf
from object_detection.utils.bbox_transform import decode_bbox, encode_bbox
from object_detection.utils.bbox_tf import pairwise_iou
from object_detection.model.losses import cls_loss, smooth_l1_loss

layers = tf.keras.layers


class RPNHead(tf.keras.Model):
    def __init__(self, num_anchors):
        super().__init__()
        self._rpn_conv = layers.Conv2D(512, [3, 3], activation=tf.nn.relu,
                                       padding='same', name='rpn_first_conv')

        self._score_num = num_anchors * 2
        self._rpn_score_conv = layers.Conv2D(self._score_num, [1, 1],
                                             padding='same', name='rpn_score_conv')
        self._rpn_score_reshape_layer = layers.Reshape([-1, 2])

        self._bbox_num = num_anchors * 4
        self._rpn_bbox_conv = layers.Conv2D(self._bbox_num, [1, 1], activation=tf.nn.relu,
                                            padding='same', name='rpn_bbox_conv')
        self._rpn_bbox_reshape_layer = layers.Reshape([-1, 4])

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        x = self._rpn_conv(inputs)
        rpn_score = self._rpn_score_conv(x)
        rpn_score_reshape = self._rpn_score_reshape_layer(rpn_score)
        rpn_score_softmax = tf.nn.softmax(rpn_score_reshape)

        rpn_bbox = self._rpn_bbox_conv(x)
        rpn_bbox_reshape = self._rpn_bbox_reshape_layer(rpn_bbox)

        return rpn_score_softmax, rpn_bbox_reshape


class RPNTrainingProposal(tf.keras.Model):
    def __init__(self,
                 pos_iou_threshold=0.7,
                 neg_iou_threshold=0.3,
                 total_num_samples=256,
                 max_pos_samples=128,):
        super().__init__()

        self._pos_iou_threshold = pos_iou_threshold
        self._neg_iou_threshold = neg_iou_threshold
        self._total_num_samples = total_num_samples
        self._max_pos_samples = max_pos_samples

    def call(self, inputs, training=None, mask=None):
        """
        生成训练rpn用的训练数据
        总体过程：
        1. 计算anchors与gt_bbox（即输入数据中的bbox）的iou。
        2. 设置（iou > 0.7）或与每个gt_bbox iou最大的anchor为整理，数量最多为
        3. 设置 iou < 0.3 的anchor为反例。
        4. 设置其他anchor不参与训练。
        5. 返回参与训练anchor的ids，以及这些ids根据顺序排列的label列表
        6. 对于正例，还需要返回其对应的 gt_bbox 的编号
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        anchors, gt_bboxes = inputs

        # [anchors_size, gt_bboxes_size]
        labels = -tf.ones((anchors.shape[0],), tf.int32)
        iou = pairwise_iou(anchors, gt_bboxes)

        # 设置与任意gt_bbox的iou > pos_iou_threshold 的 anchor 为正例
        # 设置与所有gt_bbox的iou < neg_iou_threshold 的 anchor 为反例
        max_scores = tf.argmax(iou, axis=1)
        labels = tf.where(max_scores > self._pos_iou_threshold, tf.ones_like(labels), labels)
        labels = tf.where(max_scores < self._neg_iou_threshold, tf.zeros_like(labels), labels)

        # 计算正反例真实数量
        total_pos_num = tf.reduce_sum(tf.where(labels == 1, tf.ones_like(labels), tf.zeros_like(labels)))
        total_neg_num = tf.reduce_sum(tf.where(labels == 0, tf.ones_like(labels), tf.zeros_like(labels)))

        # 根据要求，修正正反例数量
        cur_pos_num = tf.minimum(total_pos_num, self._max_pos_samples)
        cur_neg_num = tf.minimum(self._total_num_samples - cur_pos_num, total_neg_num)

        # 随机选择正例和反例
        _, total_pos_index = tf.nn.top_k(max_scores, total_pos_num, sorted=False)
        _, total_neg_index = tf.nn.top_k(-max_scores, total_neg_num, sorted=False)
        pos_index = tf.gather(total_pos_index, tf.random_shuffle(tf.range(0, total_pos_num))[:cur_pos_num])
        neg_index = tf.gather(total_neg_index, tf.random_shuffle(tf.range(0, total_neg_num))[:cur_neg_num])

        # 生成最终结果
        selected_idx = tf.concat([pos_index, neg_index], axis=0)
        return selected_idx, tf.gather(labels, selected_idx), tf.gather(max_scores, selected_idx)


class RPNProposal(tf.keras.Model):
    def __init__(self,
                 num_pre_nms_train=12000,
                 num_post_nms_train=2000,
                 num_pre_nms_test=6000,
                 num_post_nms_test=2000,
                 nms_iou_threshold=0.7,):
        super().__init__()

        self._num_pre_nms_train = num_pre_nms_train
        self._num_post_nms_train = num_post_nms_train
        self._num_pre_nms_test = num_pre_nms_test
        self._num_post_nms_test = num_post_nms_test
        self._nms_iou_threshold = nms_iou_threshold

    def call(self, inputs, training=None, mask=None):
        """
        生成后续 RoiPooling 用的bboxes，返回结果直接用于roi pooling
        总体过程：
        1. 使用anchors使用rpn_pred修正，获取所有预测结果。
        2. 根据rpn_score获取num_pre_nms个anchors。
        3. 对选中anchors进行处理（边界修正、删除非常小的anchors）
        4. 进行nms。
        5. 根据rpn_score排序，获取num_post_nms个anchors作为proposal结果。
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        # [num_anchors*feature_width*feature_height, 4]
        # [num_anchors, 4]
        # [num_anchors*feature_width*feature_height,]
        bboxes_txtytwth, anchors, scores = inputs

        # 预测结果从这里面选
        # [num_anchors*feature_width*feature_height, 4]
        decoded_bboxes = decode_bbox(anchors, bboxes_txtytwth)

        num_pre_nms = self._num_pre_nms_train if training else self._num_pre_nms_test
        num_post_nms = self._num_post_nms_train if training else self._num_post_nms_test

        # pre nms top k
        cur_top_k = tf.minimum(num_pre_nms, tf.size(scores))
        _, idx = tf.nn.top_k(scores, k=cur_top_k, sorted=False)
        decoded_bboxes = tf.gather(decoded_bboxes, idx)
        scores = tf.gather(scores, idx)

        # nms & post nms top k
        cur_top_k = tf.minimum(num_post_nms, tf.size(scores))
        selected_idx = tf.image.non_max_suppression(decoded_bboxes, scores, cur_top_k,
                                                    iou_threshold=self._nms_iou_threshold)

        return tf.gather(decoded_bboxes, selected_idx), tf.gather(scores, selected_idx)


class RpnTrainingModel(tf.keras.Model):
    def __init__(self,
                 cls_loss_weight=1.0,
                 reg_loss_weight=2.0, ):
        super().__init__()
        self._cls_loss_weight = cls_loss_weight
        self._reg_loss_weight = reg_loss_weight

        self._rpn_training_proposal = RPNTrainingProposal()

    def call(self, inputs, training=None, mask=None):
        rpn_score, rpn_bboxes_txtytwth, anchors, gt_bboxes = inputs

        # RPN 训练相关
        # 获取 rpn 的训练数据
        rpn_training_idx, rpn_training_labels, rpn_training_gt_bbox_idx = self._rpn_training_proposal(anchors,
                                                                                                      gt_bboxes)

        # inputs: rpn_training_idx, rpn_training_labels & rpn_score
        rpn_training_score = tf.gather(rpn_score, rpn_training_idx)
        rpn_cls_loss = cls_loss(rpn_training_score, rpn_training_labels, weight=self._cls_loss_weight)

        # inputs: rpn_training_idx, rpn_training_labels, rpn_training_gt_bbox_idx,
        #         rpn_bboxes_txtytwth, anchors & gt_bboxes
        # 1. rpn_bboxes_txtytwth 获取的是模型预测的修正值 tx ty tw th。
        # 2. 通过 rpn_training_idx, anchors, rpn_training_gt_bbox_idx, gt_bboxes 可获取真实修正值 ttx, tty ttw tth。
        # 3. 通过 rpn_training_idx, rpn_training_labels 可设置 loss 的weight，即只计算正例的回归损失函数。
        # 4. 通过1中的预测结果以及2中的真实结果，结合3中的损失函数权重，计算smooth L1损失函数。
        selected_anchors = tf.gather(anchors, rpn_training_idx)
        selected_gt_bboxes = tf.gather(gt_bboxes, rpn_training_gt_bbox_idx)
        bboxes_txtytwth_gt = encode_bbox(selected_anchors, selected_gt_bboxes)
        bboxes_txtytwth_pred = tf.gather(rpn_bboxes_txtytwth, rpn_training_idx)
        rpn_reg_loss = smooth_l1_loss(bboxes_txtytwth_pred, bboxes_txtytwth_gt,
                                      inside_weights=rpn_training_labels,
                                      outside_weights=self._reg_loss_weight)

        return rpn_cls_loss, rpn_reg_loss
