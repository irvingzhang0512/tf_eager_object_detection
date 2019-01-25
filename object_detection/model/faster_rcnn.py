import tensorflow as tf
from object_detection.model.rpn import RPNHead
from object_detection.utils.anchors import generate_anchors_np, anchors_filter
from object_detection.utils.bbox_transform import decode_bbox

layers = tf.keras.layers


class BaseFasterRcnn(tf.keras.Model):
    def __init__(self,
                 ratios,
                 scales,

                 extractor,
                 extractor_stride,
                 ):
        """

        :param ratios:                      anchors基本信息
        :param scales:                      anchors基本信息
        :param extractor:                   参考 feature_extractor.py
        :param extractor_stride:            表示 extractor 缩小原始图片的尺寸，faster rcnn 原始论文中为16
        """
        super().__init__()
        self._extractor = extractor
        self._extractor_stride = extractor_stride
        self._ratios = ratios
        self._scales = scales

        self._rpn_head = RPNHead(len(ratios) * len(scales))

    def call(self, inputs, training=None, mask=None):
        """
        build faster r-cnn model
        :param inputs:          shape [1, height, width, 3]
        :param training:
        :param mask:
        :return:
        """
        # extract features
        # 从 backbone 中提取特征图
        # shape like [1, height/16, width/16, 512 or 1024]
        shared_features = self._extractor(inputs, training, mask)

        # RPN Net
        # shape [1, num_anchors*feature_width*feature_height, 2], [1, num_anchors*feature_width*feature_height, 4]
        rpn_score, rpn_bboxes_txtytwth = self._rpn_head(shared_features, training, mask)
        rpn_score = tf.squeeze(rpn_score, axis=0)
        rpn_bboxes_txtytwth = tf.squeeze(rpn_bboxes_txtytwth, axis=0)

        # anchors
        _, height, width, _ = inputs.get_shape().as_list()
        anchors = generate_anchors_np(self._scales, self._ratios,
                                      (height//self._extractor_stride,
                                       width//self._extractor_stride)) * self._extractor_stride

        return shared_features, anchors, rpn_score, rpn_bboxes_txtytwth


class FasterRcnnEnd2EndTrainingModel(tf.keras.Model):
    def __init__(self,
                 base_model,
                 rpn_training_model,
                 roi_training_model):
        super().__init__()
        self._base_model = base_model
        self._rpn_training_model = rpn_training_model
        self._roi_training_model = roi_training_model

    def call(self, inputs, training=None, mask=None):
        image, gt_bboxes, gt_labels = inputs
        shared_features, anchors, rpn_score, rpn_bboxes_txtytwth = self._base_model(image, training, mask)
        rpn_cls_loss, rpn_reg_loss = self._rpn_training_model((anchors, rpn_score, rpn_bboxes_txtytwth, gt_bboxes),
                                                              training, mask)
        roi_cls_loss, roi_reg_loss = self._roi_training_model((shared_features, anchors,
                                                              rpn_score, rpn_bboxes_txtytwth,
                                                              gt_bboxes, gt_labels), training, mask)
        return rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss


class FasterRcnnPredictModel(tf.keras.Model):
    def __init__(self,
                 base_model,
                 num_classes,
                 max_num_per_class,
                 max_num_per_image,
                 nms_iou_threshold=0.7, ):
        super().__init__()
        self._base_model = base_model

        self._nms_iou_threshold = nms_iou_threshold
        self._num_classes = num_classes
        self._max_num_per_image = max_num_per_image
        self._max_num_per_class = max_num_per_class

    def call(self, inputs, training=None, mask=None):
        image, gt_bboxes, gt_labels = inputs

        # base model
        shared_features, anchors, rpn_score, rpn_bboxes_txtytwth = self._base_model(image, training, mask)

        # Rpn Proposal
        rpn_proposals_bboxes, rpn_proposals_score = self._rpn_proposal((rpn_bboxes_txtytwth,
                                                                        anchors,
                                                                        rpn_score[:, 1]), training, mask)

        # ROI Pooling
        roi_features = self._roi_pooling((shared_features, rpn_proposals_bboxes / self._extractor_stride),
                                         training, mask)
        roi_score, roi_bboxes_txtytwth = self._roi_head(roi_features, training, mask)

        # get final result
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
