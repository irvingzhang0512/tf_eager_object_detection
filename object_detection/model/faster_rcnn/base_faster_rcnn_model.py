import tensorflow as tf

from object_detection.model.region_proposal import RegionProposal
from object_detection.model.anchor_target import AnchorTarget
from object_detection.model.proposal_target import ProposalTarget
from object_detection.model.roi_pooling import RoiPoolingCropAndResize
from object_detection.model.losses import smooth_l1_loss, cls_loss
from object_detection.utils.anchor_generator import generate_by_anchor_base_tf, generate_anchor_base
from object_detection.model.prediction import post_ops_prediction

__all__ = ['BaseFasterRcnn']
layers = tf.keras.layers


class BaseFasterRcnn(tf.keras.Model):
    def __init__(self,
                 # 通用参数
                 num_classes,
                 weight_decay,
                 ratios,
                 scales,
                 extractor_stride,

                 # region proposal & anchor target 通用参数
                 rpn_proposal_means,
                 rpn_proposal_stds,

                 # region proposal 参数
                 rpn_proposal_num_pre_nms_train,
                 rpn_proposal_num_post_nms_train,
                 rpn_proposal_num_pre_nms_test,
                 rpn_proposal_num_post_nms_test,
                 rpn_proposal_nms_iou_threshold,

                 # anchor target 以及相关损失函数参数
                 rpn_sigma,
                 rpn_training_pos_iou_threshold,
                 rpn_training_neg_iou_threshold,
                 rpn_training_total_num_samples,
                 rpn_training_max_pos_samples,

                 # roi head & proposal target 参数
                 roi_proposal_means,
                 roi_proposal_stds,

                 # roi pooling 参数
                 roi_pool_size,

                 # proposal target 以及相关损失函数参数
                 roi_sigma,
                 roi_training_pos_iou_threshold,
                 roi_training_neg_iou_threshold,
                 roi_training_total_num_samples,
                 roi_training_max_pos_samples,

                 # prediction 参数
                 prediction_max_objects_per_image,
                 prediction_max_objects_per_class,
                 prediction_nms_iou_threshold,
                 prediction_score_threshold,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_decay = weight_decay

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

        self._anchor_generator = generate_by_anchor_base_tf
        self._anchor_base = tf.to_float(generate_anchor_base(extractor_stride, ratios, scales))
        self._rpn_head = RpnHead(num_anchors=self._num_anchors, weight_decay=weight_decay)
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
        self._roi_pooling = RoiPoolingCropAndResize(pool_size=roi_pool_size)
        self._proposal_target = ProposalTarget(
            pos_iou_threshold=roi_training_pos_iou_threshold,
            neg_iou_threshold=roi_training_neg_iou_threshold,
            total_num_samples=roi_training_total_num_samples,
            max_pos_samples=roi_training_max_pos_samples,
            target_means=roi_proposal_means,
            target_stds=roi_proposal_stds)

        self._extractor = self._get_extractor()
        self._roi_head = self._get_roi_head()

    def _get_roi_head(self):
        raise NotImplementedError

    def _get_extractor(self):
        raise NotImplementedError

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

        anchors = self._anchor_generator(self._anchor_base, self._extractor_stride,
                                         tf.to_int32(tf.ceil(image_shape[0] / self._extractor_stride)),
                                         tf.to_int32(tf.ceil(image_shape[1] / self._extractor_stride))
                                         )

        tf.logging.debug('anchor_generator generate {} anchors'.format(anchors.shape[0]))

        rpn_score, rpn_bbox_txtytwth = self._rpn_head(shared_features, training=training)
        rois = self._rpn_proposal((rpn_bbox_txtytwth, anchors, rpn_score, image_shape),
                                  training=training)

        if training:
            # rpn loss
            rpn_labels, rpn_bbox_targets, rpn_in_weights, rpn_out_weights = self._anchor_target((gt_bboxes,
                                                                                                 image_shape,
                                                                                                 anchors,
                                                                                                 self._num_anchors),
                                                                                                training)
            rpn_cls_loss, rpn_reg_loss = self._get_rpn_loss(rpn_score, rpn_bbox_txtytwth,
                                                            rpn_labels, rpn_bbox_targets,
                                                            rpn_in_weights, rpn_out_weights)

            # roi loss
            final_rois, roi_labels, roi_bbox_target, roi_in_weights, roi_out_weights = self._proposal_target((rois,
                                                                                                              gt_bboxes,
                                                                                                              gt_labels,
                                                                                                              ),
                                                                                                             training)
            # 训练时，只计算 proposal target 的 roi_features，一般只有128个
            roi_features = self._roi_pooling((shared_features, final_rois, self._extractor_stride),
                                             training=training)
            roi_score, roi_bboxes_txtytwth = self._roi_head(roi_features, training=training)
            roi_cls_loss, roi_reg_loss = self._get_roi_loss(roi_score, roi_bboxes_txtytwth,
                                                            roi_labels, roi_bbox_target,
                                                            roi_in_weights, roi_out_weights)
            return rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss
        else:
            # 预测时，计算所有 region proposal 生成的 roi 的 roi_features，默认为300个
            roi_features = self._roi_pooling((shared_features, rois, self._extractor_stride),
                                             training=training)
            roi_score, roi_bboxes_txtytwth = self._roi_head(roi_features, training=training)

            # pred_bboxes, pred_labels, pred_scores
            roi_score_softmax = tf.nn.softmax(roi_score)
            roi_bboxes_txtytwth = tf.reshape(roi_bboxes_txtytwth, [-1, self.num_classes, 4])
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

    def _get_rpn_loss(self, rpn_score, rpn_bbox_txtytwth,
                      anchor_target_labels, anchor_target_bboxes_txtytwth,
                      anchor_target_in_weights, anchor_target_out_weights):
        rpn_score = tf.reshape(tf.transpose(tf.reshape(rpn_score, [-1, 2, self._num_anchors]), (0, 2, 1)), [-1, 2])
        rpn_selected = tf.where(anchor_target_labels >= 0)[:, 0]
        rpn_score_selected = tf.gather(rpn_score, rpn_selected)
        rpn_labels_selected = tf.gather(anchor_target_labels, rpn_selected)
        rpn_cls_loss = cls_loss(logits=rpn_score_selected, labels=rpn_labels_selected)

        rpn_reg_loss = smooth_l1_loss(rpn_bbox_txtytwth, anchor_target_bboxes_txtytwth,
                                      anchor_target_in_weights, anchor_target_out_weights, self._rpn_sigma,
                                      dim=[0, 1])
        return rpn_cls_loss, rpn_reg_loss

    def _get_roi_loss(self, roi_score, roi_bbox_txtytwth,
                      proposal_target_labels, proposal_target_bboxes_txtytwth,
                      proposal_target_in_weights, proposal_target_out_weights):
        roi_cls_loss = cls_loss(logits=roi_score,
                                labels=proposal_target_labels)

        roi_reg_loss = smooth_l1_loss(roi_bbox_txtytwth, proposal_target_bboxes_txtytwth,
                                      proposal_target_in_weights, proposal_target_out_weights,
                                      sigma=self._roi_sigma)

        return roi_cls_loss, roi_reg_loss

    def predict_rpn(self, image, gt_bboxes):

        image_shape = image.get_shape().as_list()[1:3]
        tf.logging.debug('image shape is {}'.format(image_shape))

        shared_features = self._extractor(image, training=True)
        shared_features_shape = shared_features.get_shape().as_list()[1:3]
        tf.logging.debug('shared_features shape is {}'.format(shared_features_shape))

        anchors = generate_by_anchor_base_tf(self._anchor_base, self._extractor_stride,
                                             tf.to_int32(tf.ceil(image_shape[0] / self._extractor_stride)),
                                             tf.to_int32(tf.ceil(image_shape[1] / self._extractor_stride))
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
                                             tf.to_int32(tf.ceil(image_shape[0] / self._extractor_stride)),
                                             tf.to_int32(tf.ceil(image_shape[1] / self._extractor_stride))
                                             )
        tf.logging.debug('anchor_generator generate {} anchors'.format(anchors.shape[0]))

        rpn_score, rpn_bbox_txtytwth = self._rpn_head(shared_features, training=True)
        rois = self._rpn_proposal((rpn_bbox_txtytwth, anchors, rpn_score, image_shape, self._extractor_stride),
                                  training=True)
        # final_rois, final_labels, final_bbox_targets, bbox_inside_weights, bbox_outside_weights
        return self._proposal_target((rois, gt_bboxes, gt_labels), True)

    def test_one_image(self, img_path, min_size=600, max_size=1000, preprocessing_type='caffe'):
        from dataset.utils.tf_dataset_utils import preprocessing_func
        import numpy as np
        img = tf.image.decode_jpeg(tf.io.read_file(img_path))
        h, w = img.shape[:2]
        img = tf.reshape(img, [1, h, w, 3])
        preprocessed_image, _, _, _ = preprocessing_func(img, np.zeros([1, 4], np.float32), h, w, None,
                                                         min_size=min_size, max_size=max_size,
                                                         preprocessing_type=preprocessing_type)
        bboxes, labels, scores = self(preprocessed_image, False)
        return bboxes, labels, scores

    def im_detect(self, image, img_scale):
        image_shape = image.get_shape().as_list()[1:3]
        tf.logging.debug('image shape is {}'.format(image_shape))

        shared_features = self._extractor(image, training=False)
        shared_features_shape = shared_features.get_shape().as_list()[1:3]
        tf.logging.debug('shared_features shape is {}'.format(shared_features_shape))

        anchors = self._anchor_generator(self._anchor_base, self._extractor_stride,
                                         tf.to_int32(tf.ceil(image_shape[0] / self._extractor_stride)),
                                         tf.to_int32(tf.ceil(image_shape[1] / self._extractor_stride))
                                         )

        tf.logging.debug('anchor_generator generate {} anchors'.format(anchors.shape[0]))

        rpn_score, rpn_bbox_txtytwth = self._rpn_head(shared_features, training=False)
        rois = self._rpn_proposal((rpn_bbox_txtytwth, anchors, rpn_score, image_shape, self._extractor_stride),
                                  training=False)

        roi_features = self._roi_pooling((shared_features, rois, self._extractor_stride),
                                         training=False)
        roi_score, roi_bboxes_txtytwth = self._roi_head(roi_features, training=False)
        roi_score_softmax = tf.nn.softmax(roi_score)
        rois = rois / tf.to_float(img_scale)

        return roi_score_softmax, roi_bboxes_txtytwth, rois


class RpnHead(tf.keras.Model):
    def __init__(self, num_anchors, weight_decay=0.0001):
        """
        :param num_anchors:
        """
        super().__init__()
        self._name = 'rpn_head'
        self._num_anchors = num_anchors
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
                                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                            )

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        x = self._rpn_conv(inputs)

        rpn_score = self._rpn_score_conv(x)
        rpn_score_reshape = tf.reshape(rpn_score, [-1, self._num_anchors * 2])

        rpn_bbox = self._rpn_bbox_conv(x)
        rpn_bbox_reshape = tf.reshape(rpn_bbox, [-1, 4])

        return rpn_score_reshape, rpn_bbox_reshape
