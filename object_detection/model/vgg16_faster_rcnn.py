import tensorflow as tf

from object_detection.model.extractor.feature_extractor import Vgg16Extractor
from object_detection.model.rpn import RegionProposal, AnchorTarget, RPNHead
from object_detection.model.roi import ProposalTarget, RoiPooling, RoiHead
from object_detection.model.losses import smooth_l1_loss, cls_loss
from object_detection.utils.anchors import generate_by_anchor_base_tf, generate_anchor_base
from object_detection.model.post_ops import post_ops_prediction

__all__ = ['Vgg16FasterRcnn']


class Vgg16FasterRcnn(tf.keras.Model):
    def __init__(self,
                 num_classes=21,
                 weight_decay=0.0001,

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
        self._anchor_generator = generate_by_anchor_base_tf
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

        anchors = self._anchor_generator(self._anchor_base, self._extractor_stride,
                                         tf.to_int32(tf.ceil(image_shape[0] / self._extractor_stride)),
                                         tf.to_int32(tf.ceil(image_shape[1] / self._extractor_stride))
                                         )

        tf.logging.debug('anchor_generator generate {} anchors'.format(anchors.shape[0]))

        rpn_score, rpn_bbox_txtytwth = self._rpn_head(shared_features, training=training)
        rois = self._rpn_proposal((rpn_bbox_txtytwth, anchors, rpn_score, image_shape, self._extractor_stride),
                                  training=training)

        if training:
            # rpn loss
            rpn_labels, rpn_bbox_targets, rpn_in_weights, rpn_out_weights = self._anchor_target((gt_bboxes,
                                                                                                 image_shape,
                                                                                                 anchors,
                                                                                                 self._num_anchors),
                                                                                                True)
            rpn_cls_loss, rpn_reg_loss = self._get_rpn_loss(rpn_score, rpn_bbox_txtytwth,
                                                            rpn_labels, rpn_bbox_targets,
                                                            rpn_in_weights, rpn_out_weights)

            # roi loss
            final_rois, roi_labels, roi_bbox_target, roi_in_weights, roi_out_weights = self._proposal_target((rois,
                                                                                                              gt_bboxes,
                                                                                                              gt_labels,
                                                                                                              ),
                                                                                                             True)
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
            roi_bboxes_txtytwth = tf.reshape(roi_bboxes_txtytwth, [-1, self._num_classes, 4])
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
        roi_training_idx, roi_gt_labels, _, roi_pos_num = self._proposal_target((rois, gt_bboxes,
                                                                                 gt_labels, image_shape), True)
        return tf.gather(rois, roi_training_idx[:roi_pos_num]), roi_gt_labels[:roi_pos_num]

    def load_tf_faster_rcnn_tf_weights(self, ckpt_file_path):
        reader = tf.train.load_checkpoint(ckpt_file_path)
        extractor = self.get_layer('vgg16')
        extractor_dict = {
            "vgg_16/conv1/conv1_1/": "block1_conv1",
            "vgg_16/conv1/conv1_2/": "block1_conv2",

            "vgg_16/conv2/conv2_1/": "block2_conv1",
            "vgg_16/conv2/conv2_2/": "block2_conv2",

            "vgg_16/conv3/conv3_1/": "block3_conv1",
            "vgg_16/conv3/conv3_2/": "block3_conv2",
            "vgg_16/conv3/conv3_3/": "block3_conv3",

            "vgg_16/conv4/conv4_1/": "block4_conv1",
            "vgg_16/conv4/conv4_2/": "block4_conv2",
            "vgg_16/conv4/conv4_3/": "block4_conv3",

            "vgg_16/conv5/conv5_1/": "block5_conv1",
            "vgg_16/conv5/conv5_2/": "block5_conv2",
            "vgg_16/conv5/conv5_3/": "block5_conv3",
        }
        for slim_tensor_name_pre in extractor_dict.keys():
            extractor.get_layer(name=extractor_dict[slim_tensor_name_pre]).set_weights([
                reader.get_tensor(slim_tensor_name_pre + 'weights'),
                reader.get_tensor(slim_tensor_name_pre + 'biases'),
            ])
            tf.logging.debug('successfully loaded weights for {}'.format(extractor_dict[slim_tensor_name_pre]))

        rpn_head = self.get_layer('rpn_head')
        rpn_head_dict = {
            'vgg_16/rpn_conv/3x3/': 'rpn_first_conv',
            'vgg_16/rpn_cls_score/': 'rpn_score_conv',
            'vgg_16/rpn_bbox_pred/': 'rpn_bbox_conv',
        }
        for slim_tensor_name_pre in rpn_head_dict.keys():
            rpn_head.get_layer(rpn_head_dict[slim_tensor_name_pre]).set_weights([
                reader.get_tensor(slim_tensor_name_pre + 'weights'),
                reader.get_tensor(slim_tensor_name_pre + 'biases')
            ])
            tf.logging.debug('successfully loaded weights for {}'.format(rpn_head_dict[slim_tensor_name_pre]))

        roi_head = self.get_layer('roi_head')
        roi_head_dict = {
            'vgg_16/fc6/': 'fc1',
            'vgg_16/fc7/': 'fc2',
            'vgg_16/bbox_pred/': 'roi_head_bboxes',
            'vgg_16/cls_score/': 'roi_head_score'
        }
        for slim_tensor_name_pre in roi_head_dict.keys():
            roi_head.get_layer(roi_head_dict[slim_tensor_name_pre]).set_weights([
                reader.get_tensor(slim_tensor_name_pre + 'weights'),
                reader.get_tensor(slim_tensor_name_pre + 'biases')
            ])
            tf.logging.debug('successfully loaded weights for {}'.format(roi_head_dict[slim_tensor_name_pre]))

    def test_one_image(self, img_path, min_size=600, max_size=1000, preprocessing_type='caffe'):
        from object_detection.dataset.tf_dataset_utils import preprocessing_func
        import numpy as np
        img = tf.image.decode_jpeg(tf.io.read_file(img_path))
        h, w = img.shape[:2]
        img = tf.reshape(img, [1, h, w, 3])
        preprocessed_image, _, _, _ = preprocessing_func(img, np.zeros([1, 4], np.float32), h, w, None, None,
                                                         min_size=min_size, max_size=max_size,
                                                         preprocessing_type=preprocessing_type)
        bboxes, labels, scores = self(preprocessed_image, False)
        return bboxes, labels, scores

    def test_rois(self, rois, shared_features):
        roi_features = self._roi_pooling((shared_features, rois, self._extractor_stride),
                                         training=False)
        roi_score, roi_bboxes_txtytwth = self._roi_head(roi_features, training=False)
        return roi_features, roi_score, roi_bboxes_txtytwth

    def test_shared_features(self, image):
        return self._extractor(image)

    def test_rpn_head(self, shared_features, image_shape):
        anchors = generate_by_anchor_base_tf(self._anchor_base, self._extractor_stride,
                                             tf.to_int32(tf.ceil(image_shape[0] / self._extractor_stride)),
                                             tf.to_int32(tf.ceil(image_shape[1] / self._extractor_stride))
                                             )
        rpn_score, rpn_bbox_txtytwth = self._rpn_head(shared_features, training=False)
        rois = self._rpn_proposal((rpn_bbox_txtytwth, anchors, rpn_score, image_shape, self._extractor_stride),
                                  training=False)
        return rpn_score, rpn_bbox_txtytwth, rois

    def test_anchors(self, image_shape):
        return generate_by_anchor_base_tf(self._anchor_base, self._extractor_stride,
                                          tf.to_int32(tf.ceil(image_shape[0] / self._extractor_stride)),
                                          tf.to_int32(tf.ceil(image_shape[1] / self._extractor_stride))
                                          )

    def test_all_nets(self, image):

        image_shape = image.get_shape().as_list()[1:3]
        tf.logging.debug('image shape is {}'.format(image_shape))

        shared_features = self._extractor(image, training=False)
        shared_features_shape = shared_features.get_shape().as_list()[1:3]
        tf.logging.debug('shared_features shape is {}'.format(shared_features_shape))

        # anchors = self._anchor_generator(shape=shared_features_shape,
        #                                  ratios=self._ratios, scales=self._scales) * self._extractor_stride
        anchors = generate_by_anchor_base_tf(self._anchor_base, self._extractor_stride,
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

        return shared_features, anchors, rpn_score, rpn_bbox_txtytwth, rois, roi_features, roi_score, roi_bboxes_txtytwth

    def get_anchor_target_inputs(self, image, gt_bboxes):
        image_shape = image.get_shape().as_list()[1:3]
        tf.logging.debug('image shape is {}'.format(image_shape))

        shared_features = self._extractor(image, training=True)
        shared_features_shape = shared_features.get_shape().as_list()[1:3]
        tf.logging.debug('shared_features shape is {}'.format(shared_features_shape))

        anchors = self._anchor_generator(self._anchor_base, self._extractor_stride,
                                         tf.to_int32(tf.ceil(image_shape[0] / self._extractor_stride)),
                                         tf.to_int32(tf.ceil(image_shape[1] / self._extractor_stride))
                                         )

        tf.logging.debug('anchor_generator generate {} anchors'.format(anchors.shape[0]))

        rpn_score, rpn_bbox_txtytwth = self._rpn_head(shared_features, training=True)
        return rpn_score, gt_bboxes, image_shape, anchors, self._num_anchors

    def get_proposal_target_inputs(self, image):
        image_shape = image.get_shape().as_list()[1:3]
        tf.logging.debug('image shape is {}'.format(image_shape))

        shared_features = self._extractor(image, training=True)
        shared_features_shape = shared_features.get_shape().as_list()[1:3]
        tf.logging.debug('shared_features shape is {}'.format(shared_features_shape))

        anchors = self._anchor_generator(self._anchor_base, self._extractor_stride,
                                         tf.to_int32(tf.ceil(image_shape[0] / self._extractor_stride)),
                                         tf.to_int32(tf.ceil(image_shape[1] / self._extractor_stride))
                                         )

        tf.logging.debug('anchor_generator generate {} anchors'.format(anchors.shape[0]))

        rpn_score, rpn_bbox_txtytwth = self._rpn_head(shared_features, training=True)
        rois = self._rpn_proposal((rpn_bbox_txtytwth, anchors, rpn_score, image_shape, self._extractor_stride),
                                  training=True)
        roi_features = self._roi_pooling((shared_features, rois, self._extractor_stride),
                                         training=True)
        roi_score, roi_bboxes_txtytwth = self._roi_head(roi_features, training=False)
        return rois, roi_score, roi_bboxes_txtytwth

    def get_roi_loss_by_rpn_rois(self, image, rois, gt_bboxes, gt_labels):
        shared_features = self._extractor(image)

        # roi loss
        final_rois, roi_labels, roi_bbox_target, roi_in_weights, roi_out_weights = self._proposal_target((rois,
                                                                                                          gt_bboxes,
                                                                                                          gt_labels,
                                                                                                          ),
                                                                                                         True)
        # 训练时，只计算 proposal target 的 roi_features，一般只有128个
        roi_features = self._roi_pooling((shared_features, final_rois, self._extractor_stride),
                                         training=True)
        roi_score, roi_bboxes_txtytwth = self._roi_head(roi_features, training=False)
        roi_cls_loss, roi_reg_loss = self._get_roi_loss(roi_score, roi_bboxes_txtytwth,
                                                        roi_labels, roi_bbox_target,
                                                        roi_in_weights, roi_out_weights)
        return roi_cls_loss, roi_reg_loss, final_rois, roi_labels, roi_bbox_target, roi_in_weights, roi_out_weights

    def get_rpn_loss_by_rois(self, image, rpn_labels, rpn_bbox_targets, rpn_in_weights, rpn_out_weights):
        shared_features = self._extractor(image)
        rpn_score, rpn_bbox_txtytwth = self._rpn_head(shared_features, training=True)
        rpn_labels = tf.reshape(rpn_labels, [-1, 18])
        rpn_bbox_targets = tf.reshape(rpn_bbox_targets, [-1, 36])
        rpn_in_weights = tf.reshape(rpn_in_weights, [-1, 36])
        rpn_out_weights = tf.reshape(rpn_out_weights, [-1, 36])
        rpn_cls_loss, rpn_reg_loss = self._get_rpn_loss(rpn_score, rpn_bbox_txtytwth,
                                                        rpn_labels, rpn_bbox_targets,
                                                        rpn_in_weights, rpn_out_weights)
        return rpn_cls_loss, rpn_reg_loss

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

        return roi_score_softmax, roi_bboxes_txtytwth, rois
