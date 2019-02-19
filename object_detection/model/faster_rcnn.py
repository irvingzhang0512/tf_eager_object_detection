import tensorflow as tf
from object_detection.model.rpn import RPNHead, RPNTrainingProposal, RPNProposal
from object_detection.model.roi import RoiTrainingProposal, RoiHead, RoiPooling
from object_detection.utils.anchors import generate_anchor_base, generate_by_anchor_base_np
from object_detection.model.losses import get_rpn_loss, get_roi_loss
from tensorflow.python.platform import tf_logging

layers = tf.keras.layers

__all__ = ['BaseRoiModel', 'BaseRpnModel', 'RpnTrainingModel', 'RoiTrainingModel',
           'BaseFasterRcnnModel', 'FasterRcnnTrainingModel', ]


class BaseRpnModel(tf.keras.Model):
    def __init__(self,
                 ratios,
                 scales,
                 extractor,
                 extractor_stride,

                 weight_decay=0.0005,
                 rpn_proposal_num_pre_nms_train=12000,
                 rpn_proposal_num_post_nms_train=2000,
                 rpn_proposal_num_pre_nms_test=6000,
                 rpn_proposal_num_post_nms_test=300,
                 rpn_proposal_nms_iou_threshold=0.7,

                 num_classes=21,

                 ):
        super().__init__()

        self._weight_decay = weight_decay
        self._extractor = extractor
        self._extractor_stride = extractor_stride

        self._rpn_head = RPNHead(len(ratios) * len(scales),
                                 weight_decay=weight_decay)

        self._ratios = ratios
        self._scales = scales

        self._anchor_base = generate_anchor_base(extractor_stride, ratios=ratios, anchor_scales=scales)

        self._num_classes = num_classes

        self._rpn_proposal = RPNProposal(
            num_pre_nms_train=rpn_proposal_num_pre_nms_train,
            num_post_nms_train=rpn_proposal_num_post_nms_train,
            num_pre_nms_test=rpn_proposal_num_pre_nms_test,
            num_post_nms_test=rpn_proposal_num_post_nms_test,
            nms_iou_threshold=rpn_proposal_nms_iou_threshold,
        )

    def call(self, inputs, training=None, mask=None):
        # [1, image_height, image_width, 3]
        image_shape = inputs.get_shape().as_list()[1:3]

        # 1）输入一张图片，通过 feature_extractor 提取特征；
        shared_features = self._extractor(inputs, training)
        shared_feature_shape = shared_features.get_shape().as_list()[1:3]

        # 2）获取 RPN 初步预测结果；
        # shape [num_anchors*feature_width*feature_height, 2], [num_anchors*feature_width*feature_height, 4]
        rpn_score, rpn_bboxes_txtytwth = self._rpn_head(shared_features, training, mask)

        # 3）获取 Anchors；
        # # plan A, 性能明显较差
        # anchors = generate_anchors_np(self._scales, self._ratios,
        #                               (shared_feature_shape[0], shared_feature_shape[1])) * self._extractor_stride
        # plan B，相对 plan A 性能明显提升
        # TODO 是否能通过 tf 操作构建 anchors
        anchors = generate_by_anchor_base_np(self._anchor_base, self._extractor_stride,
                                             shared_feature_shape[0], shared_feature_shape[1])

        # 4）获取 RPN Proposal 结果；
        rpn_proposals_bboxes = self._rpn_proposal((rpn_bboxes_txtytwth,
                                                   anchors, rpn_score[:, 1],
                                                   image_shape, self._extractor_stride), training, mask)

        return image_shape, anchors, rpn_score, rpn_bboxes_txtytwth, rpn_proposals_bboxes, shared_features


class BaseRoiModel(tf.keras.Model):
    def __init__(self,
                 extractor_stride=16,
                 weight_decay=0.0005,
                 roi_pool_size=7,
                 num_classes=21,
                 roi_head_keep_dropout_rate=0.5,

                 ):
        super().__init__()
        self._extractor_stride = extractor_stride
        self._num_classes = num_classes
        self._roi_pooling = RoiPooling(roi_pool_size)
        self._roi_head = RoiHead(num_classes=num_classes,
                                 keep_rate=roi_head_keep_dropout_rate,
                                 weight_decay=weight_decay)

    def call(self, inputs, training=None, mask=None):
        shared_features, rpn_proposals_bboxes = inputs

        # 5）经过ROI Pooling，并获取进一步预测结果
        roi_features = self._roi_pooling((shared_features, rpn_proposals_bboxes, self._extractor_stride),
                                         training, mask)
        roi_score, roi_bboxes_txtytwth = self._roi_head(roi_features, training, mask)

        return roi_score, roi_bboxes_txtytwth


class BaseFasterRcnnModel(tf.keras.Model):
    def __init__(self,
                 ratios,
                 scales,
                 extractor,
                 extractor_stride,

                 weight_decay=0.0005,
                 rpn_proposal_num_pre_nms_train=12000,
                 rpn_proposal_num_post_nms_train=2000,
                 rpn_proposal_num_pre_nms_test=6000,
                 rpn_proposal_num_post_nms_test=300,
                 rpn_proposal_nms_iou_threshold=0.7,

                 roi_pool_size=7,
                 num_classes=21,
                 roi_head_keep_dropout_rate=0.5,

                 ):
        super().__init__()

        self._weight_decay = weight_decay
        self._extractor = extractor
        self._extractor_stride = extractor_stride

        self._rpn_head = RPNHead(len(ratios) * len(scales),
                                 weight_decay=weight_decay)

        self._ratios = ratios
        self._scales = scales
        self._anchor_base = generate_anchor_base(extractor_stride, ratios=ratios, anchor_scales=scales)

        self._num_classes = num_classes

        self._rpn_proposal = RPNProposal(
            num_pre_nms_train=rpn_proposal_num_pre_nms_train,
            num_post_nms_train=rpn_proposal_num_post_nms_train,
            num_pre_nms_test=rpn_proposal_num_pre_nms_test,
            num_post_nms_test=rpn_proposal_num_post_nms_test,
            nms_iou_threshold=rpn_proposal_nms_iou_threshold,
        )
        self._roi_pooling = RoiPooling(roi_pool_size)
        self._roi_head = RoiHead(num_classes=num_classes,
                                 keep_rate=roi_head_keep_dropout_rate,
                                 weight_decay=weight_decay)

    def call(self, inputs, training=None, mask=None):
        """
        Faster R-CNN 基础网络

        包括以下内容：
        1）输入一张图片，通过 feature_extractor 提取特征；
        2）获取 RPN 初步预测结果；
        3）获取 Anchors；
        4）获取 RPN Proposal 结果；
        5）经过ROI Pooling，并获取进一步预测结果。

        作用：
        1. Faster R-CNN 模型的 save/load 的本质，就是对本模型的 save/load；
        2. Faster R-CNN 的其他部分，只要超参数固定，就不会有变化；
        3. 通过本模型输出，可以用于训练（计算损失函数）或预测（用于获取最终bboxes labels）结果。
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        # [1, image_height, image_width, 3]
        image_shape = inputs.get_shape().as_list()[1:3]

        # 1）输入一张图片，通过 feature_extractor 提取特征；
        shared_features = self._extractor(inputs, training)
        shared_feature_shape = shared_features.get_shape().as_list()[1:3]

        # 2）获取 RPN 初步预测结果；
        # shape [num_anchors*feature_width*feature_height, 2], [num_anchors*feature_width*feature_height, 4]
        rpn_score, rpn_bboxes_txtytwth = self._rpn_head(shared_features, training, mask)

        # 3）获取 Anchors；
        # # plan A
        # anchors = generate_anchors_np(self._scales, self._ratios,
        #                               (shared_feature_shape[0], shared_feature_shape[1])) * self._extractor_stride
        # plan B
        anchors = generate_by_anchor_base_np(self._anchor_base, self._extractor_stride,
                                             shared_feature_shape[0], shared_feature_shape[1])

        # 4）获取 RPN Proposal 结果；
        rpn_proposals_bboxes = self._rpn_proposal((rpn_bboxes_txtytwth,
                                                   anchors, rpn_score[:, 1],
                                                   image_shape, self._extractor_stride), training, mask)

        # 5）经过ROI Pooling，并获取进一步预测结果
        roi_features = self._roi_pooling((shared_features, rpn_proposals_bboxes, self._extractor_stride),
                                         training, mask)
        roi_score, roi_bboxes_txtytwth = self._roi_head(roi_features, training, mask)

        return image_shape, anchors, rpn_score, rpn_bboxes_txtytwth, rpn_proposals_bboxes, roi_score, roi_bboxes_txtytwth


class RpnTrainingModel(tf.keras.Model):
    def __init__(self,
                 sigma=3.0,
                 rpn_training_pos_iou_threshold=0.7,
                 rpn_training_neg_iou_threshold=0.3,
                 rpn_training_total_num_samples=256,
                 rpn_training_max_pos_samples=128,
                 ):
        super().__init__()

        self._sigma = sigma
        self._rpn_training_total_num_samples = rpn_training_total_num_samples
        self._rpn_training_proposal = RPNTrainingProposal(
            pos_iou_threshold=rpn_training_pos_iou_threshold,
            neg_iou_threshold=rpn_training_neg_iou_threshold,
            total_num_samples=rpn_training_total_num_samples,
            max_pos_samples=rpn_training_max_pos_samples,
        )

    def call(self, inputs, training=None, mask=None):
        """
        获取 RPN 训练网络损失函数值
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        image_shape, anchors, rpn_score, rpn_bboxes_txtytwth, gt_bboxes = inputs

        # 获取 rpn 的训练数据
        rpn_training_idx, gt_labels, gt_bboxes_txtytwth, pos_sample_number = self._rpn_training_proposal(
            (anchors, gt_bboxes, image_shape), training)
        tf_logging.debug('pos sample num is %d' % pos_sample_number)

        rpn_cls_loss, rpn_reg_loss = get_rpn_loss(rpn_score, rpn_bboxes_txtytwth,
                                                  gt_labels, gt_bboxes_txtytwth,
                                                  rpn_training_idx, pos_sample_number,
                                                  sigma=self._rpn_sigma)

        return rpn_cls_loss, rpn_reg_loss


class RoiTrainingModel(tf.keras.Model):
    def __init__(self, num_classes=21,
                 sigma=1,
                 roi_training_pos_iou_threshold=0.5,
                 roi_training_neg_iou_threshold=0.1,
                 roi_training_total_num_samples=128,
                 roi_training_max_pos_samples=32
                 ):
        super().__init__(self)

        self._sigma = sigma
        self._num_classes = num_classes
        self._roi_training_total_num_samples = roi_training_total_num_samples

        self._roi_training_proposal = RoiTrainingProposal(
            pos_iou_threshold=roi_training_pos_iou_threshold,
            neg_iou_threshold=roi_training_neg_iou_threshold,
            total_num_samples=roi_training_total_num_samples,
            max_pos_samples=roi_training_max_pos_samples,
        )

    def call(self, inputs, training=None, mask=None):
        """
        获取 ROI 训练网络损失函数值
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        image_shape, rois, roi_score, roi_bboxes_txtytwth, gt_bboxes, gt_labels = inputs

        roi_training_idx, gt_labels, gt_bboxes_txtytwth, pos_sample_num = self._roi_training_proposal(
            (rois, gt_bboxes, gt_labels, image_shape), training, mask)

        roi_cls_loss, roi_reg_loss = get_roi_loss(roi_score, roi_bboxes_txtytwth, gt_labels, gt_bboxes_txtytwth,
                                                  roi_training_idx, pos_sample_num,
                                                  self._sigma, )

        return roi_cls_loss, roi_reg_loss


class FasterRcnnTrainingModel(tf.keras.Model):
    def __init__(self, rpn_training_model, roi_training_model):
        super().__init__()
        self._rpn_training_model = rpn_training_model
        self._roi_training_model = roi_training_model

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        gt_bboxes, gt_labels, image_shape, anchors, rpn_score, rpn_txtytwth, rois, roi_score, roi_txtytwth = inputs

        # Rpn Loss
        rpn_cls_loss, rpn_reg_loss = self._rpn_training_model((image_shape, anchors,
                                                               rpn_score, rpn_txtytwth,
                                                               gt_bboxes), training, mask)

        # Roi Loss
        roi_cls_loss, roi_reg_loss = self._roi_training_model((image_shape, rois,
                                                               roi_score, roi_txtytwth,
                                                               gt_bboxes, gt_labels), training, mask)

        return rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss
