import tensorflow as tf

from object_detection.model.extractor.feature_extractor import Vgg16Extractor
from object_detection.model.rpn import RPNProposal, RPNTrainingProposal, RPNHead
from object_detection.model.roi import RoiTrainingProposal, RoiPooling, RoiHead
from object_detection.model.losses import get_rpn_loss, get_roi_loss
from object_detection.utils.anchors import generate_by_anchor_base_tf, generate_anchor_base, generate_anchors_tf
from object_detection.model.post_ops import predict_after_roi

__all__ = ['Vgg16FasterRcnn']


class Vgg16FasterRcnn(tf.keras.Model):
    def __init__(self,
                 num_classes=21,
                 weight_decay=0.0005,

                 ratios=[0.5, 1.0, 2.0],
                 scales=[8, 16, 32],
                 extractor_stride=16,

                 rpn_proposal_means=[0, 0, 0, 0],
                 rpn_proposal_stds=[1.0, 1.0, 1.0, 1.0],

                 rpn_proposal_num_pre_nms_train=12000,
                 rpn_proposal_num_post_nms_train=2000,
                 rpn_proposal_num_pre_nms_test=6000,
                 rpn_proposal_num_post_nms_test=300,
                 rpn_proposal_nms_iou_threshold=0.7,

                 rpn_sigma=3.0,
                 rpn_training_pos_iou_threshold=0.7,
                 rpn_training_neg_iou_threshold=0.3,
                 rpn_training_total_num_samples=256,
                 rpn_training_max_pos_samples=128,

                 roi_proposal_means=[0, 0, 0, 0],
                 roi_proposal_stds=[1.0, 1.0, 1.0, 1.0],

                 roi_pool_size=7,
                 roi_head_keep_dropout_rate=0.5,
                 roi_feature_size=7 * 7 * 512,

                 roi_sigma=1,
                 roi_training_pos_iou_threshold=0.5,
                 roi_training_neg_iou_threshold=0.1,
                 roi_training_total_num_samples=128,
                 roi_training_max_pos_samples=32,

                 prediction_max_objects_per_image=15,
                 prediction_max_objects_per_class=5,
                 prediction_nms_iou_threshold=0.3,
                 prediction_score_threshold=0.3,
                 ):
        super().__init__()

        self._ratios = ratios
        self._scales = scales
        self._extractor_stride = extractor_stride
        self._rpn_sigma = rpn_sigma
        self._roi_sigma = roi_sigma

        self._roi_proposal_means = roi_proposal_means
        self._roi_proposal_stds = roi_proposal_means

        self._prediction_max_objects_per_image = prediction_max_objects_per_image
        self._prediction_max_objects_per_class = prediction_max_objects_per_class
        self._prediction_nms_iou_threshold = prediction_nms_iou_threshold
        self._prediction_score_threshold = prediction_score_threshold

        self._extractor = Vgg16Extractor()

        self._rpn_head = RPNHead(len(ratios) * len(scales), weight_decay=weight_decay)
        self._anchor_generator = generate_anchors_tf
        self._anchor_base = tf.to_float(generate_anchor_base(extractor_stride, ratios, scales))
        self._rpn_proposal = RPNProposal(
            num_pre_nms_train=rpn_proposal_num_pre_nms_train,
            num_post_nms_train=rpn_proposal_num_post_nms_train,
            num_pre_nms_test=rpn_proposal_num_pre_nms_test,
            num_post_nms_test=rpn_proposal_num_post_nms_test,
            nms_iou_threshold=rpn_proposal_nms_iou_threshold,
            target_means=rpn_proposal_means,
            target_stds=rpn_proposal_stds,
        )
        self._anchor_target = RPNTrainingProposal(
            pos_iou_threshold=rpn_training_pos_iou_threshold,
            neg_iou_threshold=rpn_training_neg_iou_threshold,
            total_num_samples=rpn_training_total_num_samples,
            max_pos_samples=rpn_training_max_pos_samples,
            target_means=rpn_proposal_means,
            target_stds=rpn_proposal_stds,
        )
        self._roi_pooling = RoiPooling(pool_size=roi_pool_size)
        self._roi_head = RoiHead(num_classes,
                                 roi_feature_size=roi_feature_size,
                                 keep_rate=roi_head_keep_dropout_rate,
                                 weight_decay=weight_decay)
        self._proposal_target = RoiTrainingProposal(
            pos_iou_threshold=roi_training_pos_iou_threshold,
            neg_iou_threshold=roi_training_neg_iou_threshold,
            total_num_samples=roi_training_total_num_samples,
            max_pos_samples=roi_training_max_pos_samples,
            target_means=roi_proposal_means,
            target_stds=roi_proposal_stds)

    def call(self, inputs, training=None, mask=None):
        if training:
            image, gt_bboxes, gt_labels = inputs
        else:
            image = inputs

        image_shape = image.get_shape().as_list()[1:3]
        tf.logging.debug('image shape is {}'.format(image_shape))

        shared_features = self._extractor(image, training=training)
        shared_features_shape = shared_features.get_shape().as_list()[1:3]
        tf.logging.debug('shared_features shape is {}'.format(shared_features_shape))

        # anchors = self._anchor_generator(shape=shared_features_shape,
        #                                  ratios=self._ratios, scales=self._scales) * self._extractor_stride
        anchors = generate_by_anchor_base_tf(self._anchor_base, self._extractor_stride,
                                             shared_features_shape[0], shared_features_shape[1])

        tf.logging.debug('anchor_generator generate {} anchors'.format(anchors.shape[0]))

        rpn_score, rpn_bbox_txtytwth = self._rpn_head(shared_features, training=training)
        rois = self._rpn_proposal((rpn_bbox_txtytwth, anchors, rpn_score[:, 1], image_shape, self._extractor_stride),
                                  training=training)
        roi_features = self._roi_pooling((shared_features, rois, self._extractor_stride),
                                         training=training)
        roi_score, roi_bboxes_txtytwth = self._roi_head(roi_features, training=training)

        if training:
            rpn_training_idx, rpn_gt_labels, rpn_gt_txtytwth, rpn_pos_num = self._anchor_target((anchors,
                                                                                                 gt_bboxes,
                                                                                                 image_shape),
                                                                                                False)
            rpn_cls_loss, rpn_reg_loss = get_rpn_loss(rpn_score, rpn_bbox_txtytwth,
                                                      rpn_gt_labels, rpn_gt_txtytwth,
                                                      rpn_training_idx, rpn_pos_num,
                                                      sigma=self._rpn_sigma)

            roi_training_idx, roi_gt_labels, roi_gt_txtytwth, roi_pos_num = self._proposal_target((rois,
                                                                                                   gt_bboxes,
                                                                                                   gt_labels,
                                                                                                   image_shape),
                                                                                                  False)
            roi_cls_loss, roi_reg_loss = get_roi_loss(roi_score, roi_bboxes_txtytwth,
                                                      roi_gt_labels, roi_gt_txtytwth,
                                                      roi_training_idx, roi_pos_num,
                                                      sigma=self._roi_sigma)
            return rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss
        else:
            # pred_bboxes, pred_labels, pred_scores
            roi_score_softmax = tf.nn.softmax(roi_score)
            return predict_after_roi(roi_score_softmax, roi_bboxes_txtytwth, rois, image_shape,
                                     self._roi_proposal_means, self._roi_proposal_stds,
                                     max_num_per_class=self._prediction_max_objects_per_class,
                                     max_num_per_image=self._prediction_max_objects_per_image,
                                     nms_iou_threshold=self._prediction_nms_iou_threshold,
                                     score_threshold=self._prediction_score_threshold,
                                     extractor_stride=self._extractor_stride)

    def predict_rpn(self, image, gt_bboxes):

        image_shape = image.get_shape().as_list()[1:3]
        tf.logging.debug('image shape is {}'.format(image_shape))

        shared_features = self._extractor(image, training=True)
        shared_features_shape = shared_features.get_shape().as_list()[1:3]
        tf.logging.debug('shared_features shape is {}'.format(shared_features_shape))

        # anchors = self._anchor_generator(shape=shared_features_shape,
        #                                  ratios=self._ratios, scales=self._scales) * self._extractor_stride
        anchors = generate_by_anchor_base_tf(self._anchor_base, self._extractor_stride,
                                             shared_features_shape[0], shared_features_shape[1])

        rpn_training_idx, _, _, rpn_pos_num = self._anchor_target((anchors,
                                                                   gt_bboxes,
                                                                   image_shape),
                                                                  False)

        return tf.gather(anchors, rpn_training_idx[:rpn_pos_num])

    def predict_roi(self, image, gt_bboxes, gt_labels):
        image_shape = image.get_shape().as_list()[1:3]
        tf.logging.debug('image shape is {}'.format(image_shape))

        shared_features = self._extractor(image, training=True)
        shared_features_shape = shared_features.get_shape().as_list()[1:3]
        tf.logging.debug('shared_features shape is {}'.format(shared_features_shape))

        # anchors = self._anchor_generator(shape=shared_features_shape,
        #                                  ratios=self._ratios, scales=self._scales) * self._extractor_stride
        anchors = generate_by_anchor_base_tf(self._anchor_base, self._extractor_stride,
                                             shared_features_shape[0], shared_features_shape[1])
        tf.logging.debug('anchor_generator generate {} anchors'.format(anchors.shape[0]))

        rpn_score, rpn_bbox_txtytwth = self._rpn_head(shared_features, training=True)
        rois = self._rpn_proposal((rpn_bbox_txtytwth, anchors, rpn_score[:, 1], image_shape, self._extractor_stride),
                                  training=True)
        roi_training_idx, roi_gt_labels, _, roi_pos_num = self._proposal_target((rois, gt_bboxes,
                                                                                 gt_labels, image_shape), True)
        return tf.gather(rois, roi_training_idx[:roi_pos_num]), roi_gt_labels[:roi_pos_num]
