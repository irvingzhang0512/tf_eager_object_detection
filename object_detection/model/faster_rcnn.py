import tensorflow as tf
from object_detection.model.rpn import RPNHead, RPNProposal
from object_detection.model.roi import RoiPooling, RoiHead
from object_detection.utils.anchors import generate_anchors_np, anchors_filter
from object_detection.utils.bbox_transform import decode_bbox

layers = tf.keras.layers


class BaseFasterRcnn(tf.keras.Model):
    def __init__(self,
                 image_size,
                 num_classes,
                 ratios,
                 scales,

                 extractor,
                 extractor_stride,

                 rpn_training_model,

                 roi_pool_size,
                 num_rois,
                 roi_training_model,

                 final_prediction_model,
                 ):
        """

        :param image_size:                  图片尺寸，(height, width)结构
        :param num_classes:                 类别数量，包括背景。如VOC取值21，COCO取值81
        :param ratios:                      anchors基本信息
        :param scales:                      anchors基本信息
        :param extractor:                   参考 feature_extractor.py
        :param extractor_stride:            表示 extractor 缩小原始图片的尺寸，faster rcnn 原始论文中为16
        :param rpn_training_model:          参考 rpn.py 中的 RpnTrainingModel
        :param roi_pool_size:               roi pooling 基本参数
        :param num_rois:                    roi pooling 基本参数
        :param roi_training_model:          参考 roi.py 中的 RoiTrainingProposal
        :param final_prediction_model:      参考 faster_rcnn.py 中的 FasterRcnnPredictModel
        """
        super().__init__()
        self._extractor = extractor
        self._extractor_stride = extractor_stride
        self._ratios = ratios
        self._scales = scales
        self._roi_pool_size = roi_pool_size
        self._num_rois = num_rois
        self._rpn_training_model = rpn_training_model
        self._roi_training_model = roi_training_model
        self._final_prediction_model = final_prediction_model

        self._anchors = generate_anchors_np(scales, ratios, image_size, 1, extractor_stride)
        self._anchors = anchors_filter(self._anchors, 0, image_size[0], 10.0)

        self._rpn_head = RPNHead(len(ratios) * len(scales))
        self._rpn_proposal = RPNProposal()

        self._roi_pooling = RoiPooling(pool_size=roi_pool_size)
        self._roi_head = RoiHead(num_classes)

    def call(self, inputs, training=None, mask=None):
        """
        build faster r-cnn model
        :param inputs:          shape [1, height, width, 3], [num_bboxes, 4], [num_bboxes,]
        :param training:
        :param mask:
        :return:
        """
        # [1, height, width, 3], [1, num_bboxes, 4], [1, num_bboxes,]
        image, gt_bboxes, gt_labels = inputs
        gt_bboxes = tf.squeeze(gt_bboxes * 384, axis=0)
        gt_labels = tf.squeeze(gt_labels, axis=0)

        # extract features
        # 从 backbone 中提取特征图
        # shape like [1, height/16, width/16, 512 or 1024]
        shared_features = self._extractor(image, training, mask)

        # RPN Net
        # shape [1, num_anchors*feature_width*feature_height, 2], [1, num_anchors*feature_width*feature_height, 4]
        rpn_score, rpn_bboxes_txtytwth = self._rpn_head(shared_features, training, mask)
        rpn_score = tf.squeeze(rpn_score, axis=0)
        rpn_bboxes_txtytwth = tf.squeeze(rpn_bboxes_txtytwth, axis=0)

        # Rpn training
        rpn_cls_loss, rpn_reg_loss = self._rpn_training_model((rpn_score,
                                                               rpn_bboxes_txtytwth,
                                                               self._anchors,
                                                               gt_bboxes), training, mask)

        # Rpn Proposal
        rpn_proposals_bboxes, rpn_proposals_score = self._rpn_proposal((rpn_bboxes_txtytwth,
                                                                        self._anchors,
                                                                        rpn_score[:, 1]), training, mask)

        # ROI Pooling
        roi_features = self._roi_pooling((shared_features, rpn_proposals_bboxes / self._extractor_stride), training,
                                         mask)
        roi_score, roi_bboxes_txtytwth = self._roi_head(roi_features, training, mask)

        # Roi Training
        roi_cls_loss, roi_reg_loss = self._roi_training_model((rpn_proposals_bboxes,
                                                               roi_score, roi_bboxes_txtytwth,
                                                               gt_bboxes, gt_labels), training, mask)

        # predicting
        final_bboxes, final_cls = self._final_prediction_model((roi_bboxes_txtytwth, roi_score, rpn_proposals_bboxes),
                                                               training, mask)

        return [rpn_cls_loss, rpn_reg_loss], [roi_cls_loss, roi_reg_loss], [final_bboxes, final_cls]


class FasterRcnnPredictModel(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 max_num_per_class,
                 max_num_per_image,
                 nms_iou_threshold=0.7, ):
        super().__init__()
        self._nms_iou_threshold = nms_iou_threshold
        self._num_classes = num_classes
        self._max_num_per_image = max_num_per_image
        self._max_num_per_class = max_num_per_class

    def call(self, inputs, training=None, mask=None):
        roi_bboxes_txtytwth, roi_score, rpn_proposals_bboxes = inputs
        final_bboxes = decode_bbox(rpn_proposals_bboxes.numpy(), roi_bboxes_txtytwth.numpy())

        res_scores = []
        res_bboxes = []
        res_cls = []
        for i in range(self._num_classes):
            cur_cls_score = roi_score[:, i]
            cur_idx = tf.image.non_max_suppression(final_bboxes, cur_cls_score,
                                                   self._max_num_per_class,
                                                   self._nms_iou_threshold)
            res_scores.append(tf.gather(cur_cls_score, cur_idx))
            res_bboxes.append(tf.gather(final_bboxes, cur_idx))
            res_cls.append(tf.ones_like(cur_idx, dtype=tf.int32) * i)

        scores_after_nms = tf.concat(res_scores, axis=0)
        bboxes_after_nms = tf.concat(res_bboxes, axis=0)
        cls_after_nms = tf.concat(res_cls, axis=0)

        _, final_idx = tf.nn.top_k(scores_after_nms, k=self._max_num_per_image, sorted=False)
        return tf.gather(bboxes_after_nms, final_idx), tf.gather(cls_after_nms, final_idx)
