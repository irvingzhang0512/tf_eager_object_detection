import tensorflow as tf

from object_detection.model.extractor.feature_extractor import Vgg16Extractor
from object_detection.model.rpn import RegionProposal, AnchorTarget, RPNHead
from object_detection.model.roi import ProposalTarget, RoiPooling, RoiHead
from object_detection.model.losses import get_rpn_loss, get_roi_loss
from object_detection.utils.anchors import generate_by_anchor_base_tf, generate_anchor_base, generate_anchors_tf
from object_detection.model.post_ops import predict_after_roi, post_ops_prediction

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
        self._num_classes = num_classes

        self._ratios = ratios
        self._scales = scales
        self._num_anchors = len(ratios) * len(scales)
        self._extractor_stride = extractor_stride
        self._rpn_sigma = rpn_sigma
        self._roi_sigma = roi_sigma

        self._roi_proposal_means = roi_proposal_means
        self._roi_proposal_stds = roi_proposal_stds

        self._prediction_max_objects_per_image = prediction_max_objects_per_image
        self._prediction_max_objects_per_class = prediction_max_objects_per_class
        self._prediction_nms_iou_threshold = prediction_nms_iou_threshold
        self._prediction_score_threshold = prediction_score_threshold

        self._extractor = Vgg16Extractor()

        self._rpn_head = RPNHead(self._num_anchors, weight_decay=weight_decay)
        self._anchor_generator = generate_anchors_tf
        self._anchor_base = tf.to_float(generate_anchor_base(extractor_stride, ratios, scales))
        self._rpn_proposal = RegionProposal(
            num_anchors=self._num_anchors,
            num_pre_nms_train=rpn_proposal_num_pre_nms_train,
            num_post_nms_train=rpn_proposal_num_post_nms_train,
            num_pre_nms_test=rpn_proposal_num_pre_nms_test,
            num_post_nms_test=rpn_proposal_num_post_nms_test,
            nms_iou_threshold=rpn_proposal_nms_iou_threshold,
            target_means=rpn_proposal_means,
            target_stds=rpn_proposal_stds,
        )
        self._anchor_target = AnchorTarget(
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
        self._proposal_target = ProposalTarget(
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

        anchors = generate_by_anchor_base_tf(self._anchor_base, self._extractor_stride,
                                             tf.to_int32(image_shape[0] / self._extractor_stride),
                                             tf.to_int32(image_shape[1] / self._extractor_stride)
                                             )

        tf.logging.debug('anchor_generator generate {} anchors'.format(anchors.shape[0]))

        rpn_score, rpn_bbox_txtytwth = self._rpn_head(shared_features, training=training)
        rois = self._rpn_proposal((rpn_bbox_txtytwth, anchors, rpn_score, image_shape, self._extractor_stride),
                                  training=training)
        roi_features = self._roi_pooling((shared_features, rois, self._extractor_stride),
                                         training=training)
        roi_score, roi_bboxes_txtytwth = self._roi_head(roi_features, training=training)

        if training:
            rpn_training_idx, rpn_gt_labels, rpn_gt_txtytwth, rpn_pos_num = self._anchor_target((anchors,
                                                                                                 gt_bboxes,
                                                                                                 image_shape),
                                                                                                False)
            # [num_anchors, 2 * num_anchors]
            rpn_score = tf.reshape(rpn_score, [-1, 2])
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
            pred_rois, pred_labels, pred_scores = post_ops_prediction(roi_score_softmax, roi_bboxes_txtytwth,
                                                                      rois, image_shape,
                                                                      self._roi_proposal_means, self._roi_proposal_stds,
                                                                      max_num_per_class=self._prediction_max_objects_per_class,
                                                                      max_num_per_image=self._prediction_max_objects_per_image,
                                                                      nms_iou_threshold=self._prediction_nms_iou_threshold,
                                                                      score_threshold=self._prediction_score_threshold,
                                                                      extractor_stride=self._extractor_stride,
                                                                      )
            return pred_rois, pred_labels, pred_scores

    def predict_rpn(self, image, gt_bboxes):

        image_shape = image.get_shape().as_list()[1:3]
        tf.logging.debug('image shape is {}'.format(image_shape))

        shared_features = self._extractor(image, training=True)
        shared_features_shape = shared_features.get_shape().as_list()[1:3]
        tf.logging.debug('shared_features shape is {}'.format(shared_features_shape))

        anchors = generate_by_anchor_base_tf(self._anchor_base, self._extractor_stride,
                                             tf.to_int32(image_shape[0] / self._extractor_stride),
                                             tf.to_int32(image_shape[1] / self._extractor_stride)
                                             )
        tf.logging.debug('generate {} anchors'.format(anchors.shape[0]))

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

        anchors = generate_by_anchor_base_tf(self._anchor_base, self._extractor_stride,
                                             tf.to_int32(image_shape[0] / self._extractor_stride),
                                             tf.to_int32(image_shape[1] / self._extractor_stride)
                                             )
        tf.logging.debug('anchor_generator generate {} anchors'.format(anchors.shape[0]))

        rpn_score, rpn_bbox_txtytwth = self._rpn_head(shared_features, training=True)
        rois = self._rpn_proposal((rpn_bbox_txtytwth, anchors, rpn_score, image_shape, self._extractor_stride),
                                  training=True)
        roi_training_idx, roi_gt_labels, _, roi_pos_num = self._proposal_target((rois, gt_bboxes,
                                                                                 gt_labels, image_shape), True)
        return tf.gather(rois, roi_training_idx[:roi_pos_num]), roi_gt_labels[:roi_pos_num]

    def _load_simple_faster_rcnn_pytorch_weights(self, pytorch_weights_dict):
        extractor = self.get_layer('vgg16')
        extractor_dict = {
            "extractor.0.": "block1_conv1",
            "extractor.2.": "block1_conv2",

            "extractor.5.": "block2_conv1",
            "extractor.7.": "block2_conv2",

            "extractor.10.": "block3_conv1",
            "extractor.12.": "block3_conv2",
            "extractor.14.": "block3_conv3",

            "extractor.17.": "block4_conv1",
            "extractor.19.": "block4_conv2",
            "extractor.21.": "block4_conv3",

            "extractor.24.": "block5_conv1",
            "extractor.26.": "block5_conv2",
            "extractor.28.": "block5_conv3",
        }
        for pytorch_name in extractor_dict.keys():
            extractor.get_layer(extractor_dict[pytorch_name]).set_weights([
                pytorch_weights_dict[pytorch_name + 'weight'],
                pytorch_weights_dict[pytorch_name + 'bias'],
            ])

        rpn_head = self.get_layer('rpn_head')
        rpn_head_dict = {
            'rpn.conv1.': 'rpn_first_conv',
            'rpn.score.': 'rpn_score_conv',
            'rpn.loc.': 'rpn_bbox_conv',
        }
        for pytorch_name in rpn_head_dict.keys():
            print(pytorch_name, rpn_head_dict[pytorch_name])
            rpn_head.get_layer(rpn_head_dict[pytorch_name]).set_weights([
                pytorch_weights_dict[pytorch_name + 'weight'],
                pytorch_weights_dict[pytorch_name + 'bias']
            ])

        roi_head = self.get_layer('roi_head')
        roi_head_dict = {
            'head.classifier.0.': 'fc1',
            'head.classifier.2.': 'fc2',
            'head.cls_loc.': 'roi_head_bboxes',
            'head.score.': 'roi_head_score'
        }
        for pytorch_name in roi_head_dict.keys():
            print(pytorch_name, roi_head_dict[pytorch_name])
            roi_head.get_layer(roi_head_dict[pytorch_name]).set_weights([
                pytorch_weights_dict[pytorch_name + 'weight'],
                pytorch_weights_dict[pytorch_name + 'bias']
            ])
