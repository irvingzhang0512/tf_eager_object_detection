import tensorflow as tf

from object_detection.model.region_proposal import RegionProposal
from object_detection.model.anchor_target import AnchorTarget
from object_detection.model.proposal_target import ProposalTarget
from object_detection.model.roi_pooling import RoiPoolingCropAndResize
from object_detection.model.losses import smooth_l1_loss, cls_loss
from object_detection.utils.anchor_generator import generate_by_anchor_base_tf, generate_anchor_base
from object_detection.model.prediction import post_ops_prediction

layers = tf.keras.layers


class BaseFPN(tf.keras.Model):
    def __init__(self,
                 # 通用参数
                 roi_feature_size=(7, 7, 256),
                 num_classes=21,
                 weight_decay=0.0001,

                 # fpn 特有参数
                 level_name_list=('p2', 'p3', 'p4', 'p5', 'p6'),
                 min_level=2,
                 max_level=5,

                 # fpn 中 anchors 特有参数
                 anchor_stride_list=(4, 8, 16, 32, 64),
                 base_anchor_size_list=(32, 64, 128, 256, 512),
                 ratios=(0.5, 1.0, 2.0),
                 scales=(1., ),

                 # region proposal & anchor target 通用参数
                 rpn_proposal_means=(0, 0, 0, 0),
                 rpn_proposal_stds=(1.0, 1.0, 1.0, 1.0),

                 # region proposal 参数
                 rpn_proposal_num_pre_nms_train=12000,
                 rpn_proposal_num_post_nms_train=2000,
                 rpn_proposal_num_pre_nms_test=6000,
                 rpn_proposal_num_post_nms_test=300,
                 rpn_proposal_nms_iou_threshold=0.7,

                 # anchor target 以及相关损失函数参数
                 rpn_sigma=3.0,
                 rpn_training_pos_iou_threshold=0.7,
                 rpn_training_neg_iou_threshold=0.3,
                 rpn_training_total_num_samples=256,
                 rpn_training_max_pos_samples=128,

                 # roi head & proposal target 参数
                 roi_proposal_means=(0, 0, 0, 0),
                 roi_proposal_stds=(0.1, 0.1, 0.2, 0.2),

                 # roi pooling 参数
                 roi_pool_size=7,

                 # proposal target 以及相关损失函数参数
                 roi_sigma=1,
                 roi_training_pos_iou_threshold=0.5,
                 roi_training_neg_iou_threshold=0.1,
                 roi_training_total_num_samples=128,
                 roi_training_max_pos_samples=32,

                 # prediction 参数
                 prediction_max_objects_per_image=50,
                 prediction_max_objects_per_class=50,
                 prediction_nms_iou_threshold=0.3,
                 prediction_score_threshold=0.,
                 ):
        super().__init__()
        # 当(extractor & roi head)以及(rpn head, region proposal, anchor target, proposal target)同时用到某参数时
        # 该参数不能作为父类的私有变量
        self.roi_feature_size = roi_feature_size
        self.num_classes = num_classes
        self.weight_decay = weight_decay

        # fpn 特有参数
        self._level_name_list = level_name_list
        self._min_level = min_level
        self._max_level = max_level

        # fpn 中 anchors 相关参数
        self._anchor_stride_list = anchor_stride_list
        self._base_anchor_size_list = base_anchor_size_list
        self._ratios = ratios
        self._scales = scales
        self._num_anchors = len(ratios) * len(scales)
        self._anchor_generator = generate_by_anchor_base_tf

        # 生成 base anchors
        self._anchor_base_list = []
        for base_size in base_anchor_size_list:
            self._anchor_base_list.append(tf.to_float(generate_anchor_base(base_size, ratios, scales)))

        # 计算损失函数所需参数
        self._rpn_sigma = rpn_sigma
        self._roi_sigma = roi_sigma

        # 预测所需参数
        self._roi_proposal_means = roi_proposal_means
        self._roi_proposal_stds = roi_proposal_stds
        self._prediction_max_objects_per_image = prediction_max_objects_per_image
        self._prediction_max_objects_per_class = prediction_max_objects_per_class
        self._prediction_nms_iou_threshold = prediction_nms_iou_threshold
        self._prediction_score_threshold = prediction_score_threshold

        # 获取 FPN 基本组件
        self._extractor = self._get_extractor()
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
        self._roi_pooling = RoiPoolingCropAndResize(pool_size=roi_pool_size)
        self._roi_head = self._get_roi_head()

        # 训练组件
        self._anchor_target = AnchorTarget(
            pos_iou_threshold=rpn_training_pos_iou_threshold,
            neg_iou_threshold=rpn_training_neg_iou_threshold,
            total_num_samples=rpn_training_total_num_samples,
            max_pos_samples=rpn_training_max_pos_samples,
            target_means=rpn_proposal_means,
            target_stds=rpn_proposal_stds,
        )
        self._proposal_target = ProposalTarget(
            pos_iou_threshold=roi_training_pos_iou_threshold,
            neg_iou_threshold=roi_training_neg_iou_threshold,
            total_num_samples=roi_training_total_num_samples,
            max_pos_samples=roi_training_max_pos_samples,
            target_means=roi_proposal_means,
            target_stds=roi_proposal_stds)

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

        # get backbone results: p2, p3, p4, p5, p6
        p_list = self._extractor(image, training=training)
        tf.logging.debug('shared_features length is {}'.format(len(p_list)))
        for idx, p in enumerate(p_list):
            tf.logging.debug('p{} shape is {}'.format(idx + 2, p.get_shape().as_list()))

        # get rpn head results
        all_fpn_scores = []
        all_fpn_bbox_pred = []
        for level_name, p in zip(self._level_name_list, p_list):
            cur_score, cur_bboxes_pred = self._rpn_head(p)
            all_fpn_scores.append(cur_score)
            all_fpn_bbox_pred.append(cur_bboxes_pred)
            tf.logging.debug('fpn {} get rpn score shape {}'.format(level_name, cur_score.get_shape().as_list()))
        all_fpn_bbox_pred = tf.concat(all_fpn_bbox_pred, axis=0, name='all_fpn_bbox_pred')
        all_fpn_scores = tf.concat(all_fpn_scores, axis=0, name='all_fpn_scores')
        tf.logging.debug('all_fpn_bbox_pred shape is {}'.format(all_fpn_bbox_pred.get_shape().as_list()))
        tf.logging.debug('all_fpn_scores shape is {}'.format(all_fpn_scores.get_shape().as_list()))

        # generate anchors
        all_anchors = []
        for idx in range(len(self._level_name_list)):
            level_name = self._level_name_list[idx]
            extractor_stride = self._anchor_stride_list[idx]
            anchor_base = self._anchor_base_list[idx]

            cur_anchors = self._anchor_generator(anchor_base, extractor_stride,
                                                 tf.to_int32(image_shape[0] / extractor_stride),
                                                 tf.to_int32(image_shape[1] / extractor_stride)
                                                 )
            all_anchors.append(cur_anchors)
            tf.logging.debug('{} generate {} anchors'.format(level_name, cur_anchors.shape[0]))
        all_anchors = tf.concat(all_anchors, axis=0, name='all_anchors')
        tf.logging.debug('all_anchors shape is {}'.format(all_anchors.get_shape().as_list()))

        # generate rpn results: rois
        rois = self._rpn_proposal((all_fpn_bbox_pred, all_anchors, all_fpn_scores, image_shape),
                                  training=training)

        if training:
            # rpn loss
            rpn_labels, rpn_bbox_targets, rpn_in_weights, rpn_out_weights = self._anchor_target((gt_bboxes,
                                                                                                 image_shape,
                                                                                                 all_anchors,
                                                                                                 self._num_anchors),
                                                                                                True)
            rpn_cls_loss, rpn_reg_loss = self._get_rpn_loss(all_fpn_scores, all_fpn_bbox_pred,
                                                            rpn_labels, rpn_bbox_targets,
                                                            rpn_in_weights, rpn_out_weights)

            # roi loss
            final_rois, roi_labels, roi_bbox_target, roi_in_weights, roi_out_weights = self._proposal_target((rois,
                                                                                                              gt_bboxes,
                                                                                                              gt_labels,
                                                                                                              ),
                                                                                                             True)
            # 训练时，只计算 proposal target 的 roi_features，一般只有128个
            all_roi_features = []
            rois_list, selected_idx = self._assign_levels(final_rois)
            roi_labels = tf.gather(roi_labels, selected_idx)
            roi_bbox_target = tf.gather(roi_bbox_target, selected_idx)
            roi_in_weights = tf.gather(roi_in_weights, selected_idx)
            roi_out_weights = tf.gather(roi_out_weights, selected_idx)

            for level_name, cur_rois, cur_p, cur_stride in zip(self._level_name_list[:-1],
                                                               rois_list, p_list, self._anchor_stride_list):
                if cur_rois.shape[0] == 0:
                    continue
                cur_roi_features = self._roi_pooling((cur_p, cur_rois, cur_stride), training=training)
                all_roi_features.append(cur_roi_features)
                tf.logging.debug('{} generate {} roi features'.format(level_name, cur_roi_features.shape[0]))
            roi_features = tf.concat(all_roi_features, axis=0, name='all_roi_features')
            roi_score, roi_bboxes_txtytwth = self._roi_head(roi_features, training=training)
            roi_cls_loss, roi_reg_loss = self._get_roi_loss(roi_score, roi_bboxes_txtytwth,
                                                            roi_labels, roi_bbox_target,
                                                            roi_in_weights, roi_out_weights)
            return rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss
        else:
            # 预测时，计算所有 region proposal 生成的 roi 的 roi_features，默认为300个
            all_roi_features = []
            rois_list, _ = self._assign_levels(rois)
            for level_name, cur_rois, cur_p, cur_stride in zip(self._level_name_list[:-1],
                                                               rois_list, p_list, self._anchor_stride_list):
                if cur_rois.shape[0] == 0:
                    continue
                cur_roi_features = self._roi_pooling((cur_p, cur_rois, cur_stride), training=training)
                all_roi_features.append(cur_roi_features)
                tf.logging.debug('{} generate {} roi features'.format(level_name, cur_roi_features.shape[0]))
            roi_features = tf.concat(all_roi_features, axis=0, name='all_roi_features')
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
                                                                      extractor_stride=16,
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

    def _assign_levels(self, all_rois):
        with tf.name_scope('assign_levels'):
            # 计算 levels
            xmin, ymin, xmax, ymax = tf.unstack(all_rois, axis=1)
            h = tf.maximum(0., ymax - ymin)
            w = tf.maximum(0., xmax - xmin)
            levels = tf.floor(4. + tf.log(tf.sqrt(w * h + 1e-8) / 224.0) / tf.log(2.))

            # 设置level上下限
            levels = tf.maximum(levels, tf.ones_like(levels) * self._min_level)
            levels = tf.minimum(levels, tf.ones_like(levels) * self._max_level)
            levels = tf.stop_gradient(tf.reshape(levels, [-1]))

            rois_list = []
            roi_index_list = []
            for i in range(self._min_level, self._max_level + 1):
                level_i_indices = tf.reshape(tf.where(tf.equal(levels, i)), [-1])
                level_i_rois = tf.gather(all_rois, level_i_indices)
                rois_list.append(level_i_rois)
                roi_index_list.append(level_i_indices)

            return rois_list, tf.concat(roi_index_list, axis=0, name='assign_level_idx')

    def predict_rpns(self, image, gt_bboxes, training=True):
        image_shape = image.get_shape().as_list()[1:3]
        tf.logging.debug('image shape is {}'.format(image_shape))

        # get backbone results: p2, p3, p4, p5, p6
        p_list = self._extractor(image, training=training)
        tf.logging.debug('shared_features length is {}'.format(len(p_list)))
        for idx, p in enumerate(p_list):
            tf.logging.debug('p{} shape is {}'.format(idx + 2, p.get_shape().as_list()))

        # get rpn head results
        all_fpn_scores = []
        all_fpn_bbox_pred = []
        for level_name, p in zip(self._level_name_list, p_list):
            cur_score, cur_bboxes_pred = self._rpn_head(p)
            all_fpn_scores.append(cur_score)
            all_fpn_bbox_pred.append(cur_bboxes_pred)
            tf.logging.debug('fpn {} get rpn score shape {}'.format(level_name, cur_score.get_shape().as_list()))
        all_fpn_bbox_pred = tf.concat(all_fpn_bbox_pred, axis=0, name='all_fpn_bbox_pred')
        all_fpn_scores = tf.concat(all_fpn_scores, axis=0, name='all_fpn_scores')
        tf.logging.debug('all_fpn_bbox_pred shape is {}'.format(all_fpn_bbox_pred.get_shape().as_list()))
        tf.logging.debug('all_fpn_scores shape is {}'.format(all_fpn_scores.get_shape().as_list()))

        # generate anchors
        all_anchors = []
        for idx in range(len(self._level_name_list)):
            level_name = self._level_name_list[idx]
            extractor_stride = self._anchor_stride_list[idx]
            anchor_base = self._anchor_base_list[idx]

            cur_anchors = self._anchor_generator(anchor_base, extractor_stride,
                                                 tf.to_int32(image_shape[0] / extractor_stride),
                                                 tf.to_int32(image_shape[1] / extractor_stride)
                                                 )
            all_anchors.append(cur_anchors)
            tf.logging.debug('{} generate {} anchors'.format(level_name, cur_anchors.shape[0]))
        all_anchors = tf.concat(all_anchors, axis=0, name='all_anchors')
        tf.logging.debug('all_anchors shape is {}'.format(all_anchors.get_shape().as_list()))

        rpn_labels, rpn_bbox_targets, rpn_in_weights, rpn_out_weights = self._anchor_target((gt_bboxes,
                                                                                             image_shape,
                                                                                             all_anchors,
                                                                                             self._num_anchors),
                                                                                            True)
        idx = tf.where(rpn_labels > 0)[:, 0]
        return tf.gather(all_anchors, idx)

    def predict_rois(self, image, gt_bboxes, gt_labels, training=True):
        image_shape = image.get_shape().as_list()[1:3]
        tf.logging.debug('image shape is {}'.format(image_shape))

        # get backbone results: p2, p3, p4, p5, p6
        p_list = self._extractor(image, training=training)
        tf.logging.debug('shared_features length is {}'.format(len(p_list)))
        for idx, p in enumerate(p_list):
            tf.logging.debug('p{} shape is {}'.format(idx + 2, p.get_shape().as_list()))

        # get rpn head results
        all_fpn_scores = []
        all_fpn_bbox_pred = []
        for level_name, p in zip(self._level_name_list, p_list):
            cur_score, cur_bboxes_pred = self._rpn_head(p)
            all_fpn_scores.append(cur_score)
            all_fpn_bbox_pred.append(cur_bboxes_pred)
            tf.logging.debug('fpn {} get rpn score shape {}'.format(level_name, cur_score.get_shape().as_list()))
        all_fpn_bbox_pred = tf.concat(all_fpn_bbox_pred, axis=0, name='all_fpn_bbox_pred')
        all_fpn_scores = tf.concat(all_fpn_scores, axis=0, name='all_fpn_scores')
        tf.logging.debug('all_fpn_bbox_pred shape is {}'.format(all_fpn_bbox_pred.get_shape().as_list()))
        tf.logging.debug('all_fpn_scores shape is {}'.format(all_fpn_scores.get_shape().as_list()))

        # generate anchors
        all_anchors = []
        for idx in range(len(self._level_name_list)):
            level_name = self._level_name_list[idx]
            extractor_stride = self._anchor_stride_list[idx]
            anchor_base = self._anchor_base_list[idx]

            cur_anchors = self._anchor_generator(anchor_base, extractor_stride,
                                                 tf.to_int32(image_shape[0] / extractor_stride),
                                                 tf.to_int32(image_shape[1] / extractor_stride)
                                                 )
            all_anchors.append(cur_anchors)
            tf.logging.debug('{} generate {} anchors'.format(level_name, cur_anchors.shape[0]))
        all_anchors = tf.concat(all_anchors, axis=0, name='all_anchors')
        tf.logging.debug('all_anchors shape is {}'.format(all_anchors.get_shape().as_list()))

        # generate rpn results: rois
        rois = self._rpn_proposal((all_fpn_bbox_pred, all_anchors, all_fpn_scores, image_shape),
                                  training=training)

        # roi loss
        final_rois, roi_labels, roi_bbox_target, roi_in_weights, roi_out_weights = self._proposal_target((rois,
                                                                                                          gt_bboxes,
                                                                                                          gt_labels,
                                                                                                          ),
                                                                                                         True)
        # 训练时，只计算 proposal target 的 roi_features，一般只有128个
        rois_list, selected_idx = self._assign_levels(final_rois)
        roi_labels = tf.gather(roi_labels, selected_idx)

        idx = tf.where(roi_labels > 0)[:, 0]
        new_rois = []
        for cur_rois_list in rois_list:
            if cur_rois_list.shape[0] != 0:
                new_rois.append(cur_rois_list)
        new_rois = tf.concat(new_rois, axis=0)
        return tf.gather(new_rois, idx)

    def im_detect(self, image, img_scale):
        image_shape = image.get_shape().as_list()[1:3]
        tf.logging.debug('image shape is {}'.format(image_shape))

        # get backbone results: p2, p3, p4, p5, p6
        p_list = self._extractor(image, training=False)
        tf.logging.debug('shared_features length is {}'.format(len(p_list)))
        for idx, p in enumerate(p_list):
            tf.logging.debug('p{} shape is {}'.format(idx + 2, p.get_shape().as_list()))

        # get rpn head results
        all_fpn_scores = []
        all_fpn_bbox_pred = []
        for level_name, p in zip(self._level_name_list, p_list):
            cur_score, cur_bboxes_pred = self._rpn_head(p, False)
            all_fpn_scores.append(cur_score)
            all_fpn_bbox_pred.append(cur_bboxes_pred)
            tf.logging.debug('fpn {} get rpn score shape {}'.format(level_name, cur_score.get_shape().as_list()))
        all_fpn_bbox_pred = tf.concat(all_fpn_bbox_pred, axis=0, name='all_fpn_bbox_pred')
        all_fpn_scores = tf.concat(all_fpn_scores, axis=0, name='all_fpn_scores')
        tf.logging.debug('all_fpn_bbox_pred shape is {}'.format(all_fpn_bbox_pred.get_shape().as_list()))
        tf.logging.debug('all_fpn_scores shape is {}'.format(all_fpn_scores.get_shape().as_list()))

        # generate anchors
        all_anchors = []
        for idx in range(len(self._level_name_list)):
            level_name = self._level_name_list[idx]
            extractor_stride = self._anchor_stride_list[idx]
            anchor_base = self._anchor_base_list[idx]

            cur_anchors = self._anchor_generator(anchor_base, extractor_stride,
                                                 tf.to_int32(tf.ceil(image_shape[0] / extractor_stride)),
                                                 tf.to_int32(tf.ceil(image_shape[1] / extractor_stride))
                                                 )
            all_anchors.append(cur_anchors)
            tf.logging.debug('{} generate {} anchors'.format(level_name, cur_anchors.shape[0]))
        all_anchors = tf.concat(all_anchors, axis=0, name='all_anchors')
        tf.logging.debug('all_anchors shape is {}'.format(all_anchors.get_shape().as_list()))

        # generate rpn results: rois
        rois = self._rpn_proposal((all_fpn_bbox_pred, all_anchors, all_fpn_scores, image_shape),
                                  training=False)

        # 预测时，计算所有 region proposal 生成的 roi 的 roi_features，默认为300个
        all_roi_features = []
        rois_list, _ = self._assign_levels(rois)
        for level_name, cur_rois, cur_p, cur_stride in zip(self._level_name_list[:-1],
                                                           rois_list, p_list, self._anchor_stride_list):
            if cur_rois.shape[0] == 0:
                continue
            cur_roi_features = self._roi_pooling((cur_p, cur_rois, cur_stride), training=False)
            all_roi_features.append(cur_roi_features)
            tf.logging.debug('{} generate {} roi features'.format(level_name, cur_roi_features.shape[0]))
        roi_features = tf.concat(all_roi_features, axis=0, name='all_roi_features')
        roi_score, roi_bboxes_txtytwth = self._roi_head(roi_features, training=False)

        # pred_bboxes, pred_labels, pred_scores
        roi_score_softmax = tf.nn.softmax(roi_score)

        new_rois = []
        for cur_rois_list in rois_list:
            if cur_rois_list.shape[0] != 0:
                new_rois.append(cur_rois_list)
        new_rois = tf.concat(new_rois, axis=0)
        new_rois = new_rois / tf.to_float(img_scale)

        return roi_score_softmax, roi_bboxes_txtytwth, new_rois


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
