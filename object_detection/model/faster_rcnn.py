import tensorflow as tf
import numpy as np
from object_detection.model.rpn import RPNHead, RPNTrainingProposal, RPNProposal, proposal_filter
from object_detection.model.roi import RoiTrainingProposal, RoiHead, RoiPooling
from object_detection.utils.anchors import generate_anchors_np
from object_detection.utils.bbox_transform import decode_bbox, encode_bbox
from object_detection.model.losses import cls_loss, smooth_l1_loss

layers = tf.keras.layers


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
        shared_features = self._extractor(inputs, training, mask)

        # 2）获取 RPN 初步预测结果；
        # shape [num_anchors*feature_width*feature_height, 2], [num_anchors*feature_width*feature_height, 4]
        rpn_score, rpn_bboxes_txtytwth = self._rpn_head(shared_features, training, mask)

        # 3）获取 Anchors；
        anchors = generate_anchors_np(self._scales, self._ratios,
                                      (image_shape[0] // self._extractor_stride,
                                       image_shape[1] // self._extractor_stride)) * self._extractor_stride

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
                 cls_loss_weight=3,
                 reg_loss_weight=1,
                 rpn_training_pos_iou_threshold=0.7,
                 rpn_training_neg_iou_threshold=0.3,
                 rpn_training_total_num_samples=256,
                 rpn_training_max_pos_samples=128, ):
        super().__init__()

        self._cls_loss_weight = cls_loss_weight
        self._reg_loss_weight = reg_loss_weight
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
        rpn_training_idx, rpn_training_labels, rpn_training_gt_bbox_idx, pos_sample_number = self._rpn_training_proposal(
            (anchors,
             gt_bboxes,
             image_shape),
            training,
            mask)

        # inputs: rpn_training_idx, rpn_training_labels & rpn_score
        rpn_training_score = tf.gather(rpn_score, rpn_training_idx)
        rpn_cls_loss = cls_loss(rpn_training_score, rpn_training_labels, weight=self._cls_loss_weight)

        # inputs: rpn_training_idx, rpn_training_labels, rpn_training_gt_bbox_idx,
        #         rpn_bboxes_txtytwth, anchors & gt_bboxes
        # 1. rpn_bboxes_txtytwth 获取的是模型预测的修正值 tx ty tw th。
        # 2. 通过 rpn_training_idx, anchors, rpn_training_gt_bbox_idx, gt_bboxes 可获取真实修正值 ttx, tty ttw tth。
        # 3. 通过 rpn_training_idx, rpn_training_labels 可设置 loss 的weight，即只计算正例的回归损失函数。
        # 4. 通过1中的预测结果以及2中的真实结果，结合3中的损失函数权重，计算smooth L1损失函数。
        # 计算损失函数中的 label，可以使用Numpy
        if pos_sample_number == 0:
            rpn_reg_loss = 0
        else:
            rpn_training_idx = rpn_training_idx[:pos_sample_number]  # [num_pos, ]
            rpn_training_gt_bbox_idx = rpn_training_gt_bbox_idx[:pos_sample_number]  # [num_pos, ]
            selected_anchors = tf.gather(anchors, rpn_training_idx)  # [num_pos, 4]
            selected_gt_bboxes = tf.gather(gt_bboxes, rpn_training_gt_bbox_idx)  # [num_pos, 4]
            bboxes_txtytwth_gt = encode_bbox(selected_anchors.numpy(), selected_gt_bboxes.numpy())  # [num_pos, 4]
            # 计算损失函数中的logits，不能使用numpy
            bboxes_txtytwth_pred = tf.gather(rpn_bboxes_txtytwth, rpn_training_idx)  # [num_pos, 4]

            # print(bboxes_txtytwth_gt)
            # print(bboxes_txtytwth_pred)
            rpn_reg_loss = smooth_l1_loss(bboxes_txtytwth_pred, bboxes_txtytwth_gt,
                                          outside_weights=self._reg_loss_weight, )

        return rpn_cls_loss, rpn_reg_loss


class RoiTrainingModel(tf.keras.Model):
    def __init__(self, num_classes=21,
                 cls_loss_weight=3,
                 reg_loss_weight=1,
                 roi_training_pos_iou_threshold=0.5,
                 roi_training_neg_iou_threshold=0.1,
                 roi_training_total_num_samples=128,
                 roi_training_max_pos_samples=32
                 ):
        super().__init__(self)

        self._cls_loss_weight = cls_loss_weight
        self._reg_loss_weight = reg_loss_weight
        self._num_classes = num_classes

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

        image_shape, rpn_proposals_bboxes, roi_score, roi_bboxes_txtytwth, gt_bboxes, gt_labels = inputs

        # 获取 roi 的训练数据
        # 不参与训练
        roi_training_idx, roi_training_labels, roi_training_gt_bbox_idx, pos_sample_num = self._roi_training_proposal(
            (rpn_proposals_bboxes, gt_bboxes, image_shape), training, mask)

        # cal roi cls loss
        roi_training_score = tf.gather(roi_score, roi_training_idx)
        roi_cls_loss = cls_loss(logits=roi_training_score,
                                labels=tf.gather(gt_labels, roi_training_gt_bbox_idx) * roi_training_labels,
                                weight=self._cls_loss_weight)

        # cal roi bbox reg loss
        # inputs: roi_training_idx, roi_training_labels, roi_training_gt_bbox_idx,
        #         roi_predicting_bboxes, rpn_proposals_bboxes, gt_bboxes
        # 只计算正例的损失函数
        if pos_sample_num == 0:
            roi_reg_loss = 0
        else:
            roi_training_idx = roi_training_idx[:pos_sample_num]  # [num_pos, ]
            roi_training_gt_bbox_idx = tf.to_int32(roi_training_gt_bbox_idx[:pos_sample_num])  # [num_pos, ]
            bboxes_txtytwth_pred = tf.reshape(tf.gather(roi_bboxes_txtytwth, roi_training_idx),
                                              [-1, self._num_classes, 4])  # [num_pos, num_classes, 4]

            # [num_pos, num_classes, 4]  [num_pos, 1]
            bboxes_txtytwth_pred = tf.squeeze(tf.batch_gather(bboxes_txtytwth_pred,
                                                              tf.expand_dims(roi_training_gt_bbox_idx, axis=-1)),
                                              axis=1)  # [num_pos, 4]

            selected_rpn_proposal_bboxes = tf.gather(rpn_proposals_bboxes, roi_training_idx)
            selected_gt_bboxes = tf.gather(gt_bboxes, roi_training_gt_bbox_idx)
            bboxes_txtytwth_gt = encode_bbox(selected_rpn_proposal_bboxes.numpy(),
                                             selected_gt_bboxes.numpy())

            roi_reg_loss = smooth_l1_loss(bboxes_txtytwth_pred, bboxes_txtytwth_gt,
                                          outside_weights=self._reg_loss_weight)

        return roi_cls_loss, roi_reg_loss


class FasterRcnnTrainingModel(tf.keras.Model):
    def __init__(self, rpn_training_model, roi_training_model):
        super().__init__()
        self._rpn_training_model = rpn_training_model
        self._roi_training_model = roi_training_model

    def call(self, inputs, training=None, mask=None):
        gt_bboxes, gt_labels, image_shape, anchors, rpn_score, rpn_bboxes_txtytwth, rpn_proposals_bboxes, roi_score, roi_bboxes_txtytwth = inputs
        gt_bboxes = tf.squeeze(gt_bboxes, axis=0)
        gt_labels = tf.to_int32(tf.squeeze(gt_labels, axis=0))

        # Rpn Loss
        rpn_cls_loss, rpn_reg_loss = self._rpn_training_model((image_shape, anchors,
                                                               rpn_score, rpn_bboxes_txtytwth,
                                                               gt_bboxes), training, mask)

        # Roi Loss
        roi_cls_loss, roi_reg_loss = self._roi_training_model((image_shape, rpn_proposals_bboxes,
                                                               roi_score, roi_bboxes_txtytwth,
                                                               gt_bboxes, gt_labels), training, mask)

        return rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss


def post_ops_prediction(inputs,
                        num_classes=21,
                        max_num_per_class=5,
                        max_num_per_image=5,
                        nms_iou_threshold=0.5,
                        ):
    rpn_proposals_bboxes, roi_score, roi_bboxes_txtytwth, image_shape = inputs
    final_bboxes = decode_bbox(rpn_proposals_bboxes.numpy(), roi_bboxes_txtytwth.numpy())
    final_bboxes, _ = proposal_filter(final_bboxes, 0, image_shape[0], image_shape[1], 16)

    res_scores = []
    res_bboxes = []
    res_cls = []
    for i in range(num_classes):
        if i == 0:
            continue
        cur_cls_score = roi_score[:, i]
        cur_idx = tf.image.non_max_suppression(final_bboxes, cur_cls_score,
                                               max_num_per_class,
                                               nms_iou_threshold)
        res_scores.append(tf.gather(cur_cls_score, cur_idx))
        res_bboxes.append(tf.gather(final_bboxes, cur_idx))
        res_cls.append(tf.ones_like(cur_idx, dtype=tf.int32) * i)

    scores_after_nms = tf.concat(res_scores, axis=0)
    bboxes_after_nms = tf.concat(res_bboxes, axis=0)
    cls_after_nms = tf.concat(res_cls, axis=0)

    _, final_idx = tf.nn.top_k(scores_after_nms, k=max_num_per_image, sorted=False)
    return tf.gather(bboxes_after_nms, final_idx), tf.gather(cls_after_nms, final_idx), tf.gather(scores_after_nms,
                                                                                                  final_idx)
