import tensorflow as tf
from object_detection.utils.bbox_tf import pairwise_iou
from tensorflow.python.platform import tf_logging
from object_detection.utils.bbox_transform import encode_bbox_with_mean_and_std

layers = tf.keras.layers

VGG_16_WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels.h5')

__all__ = ['RoiHead', 'RoiPooling', 'ProposalTarget', 'roi_align']


def crop_and_resize(image, boxes, box_ind, crop_size, pad_border=True):
    assert isinstance(crop_size, int), crop_size
    boxes = tf.stop_gradient(boxes)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        image = tf.pad(image, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
        boxes = boxes + 1

    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        """
        The way tf.image.crop_and_resize works (with normalized box):
        Initial point (the value of output[0]): x0_box * (W_img - 1)
        Spacing: w_box * (W_img - 1) / (W_crop - 1)
        Use the above grid to bilinear sample.
        However, what we want is (with fpcoor box):
        Spacing: w_box / W_crop
        Initial point: x0_box + spacing/2 - 0.5
        (-0.5 because bilinear sample (in my definition) assumes floating point coordinate
         (0.0, 0.0) is the same as pixel value (0, 0))
        This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize
        Returns:
            y1x1y2x2
        """
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.cast(crop_shape[1], tf.float32)
        spacing_h = (y1 - y0) / tf.cast(crop_shape[0], tf.float32)

        imshape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]
        nx0 = (x0 + spacing_w / 2 - 0.5) / imshape[1]
        ny0 = (y0 + spacing_h / 2 - 0.5) / imshape[0]

        nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / imshape[1]
        nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / imshape[0]

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    image_shape = tf.shape(image)[1:3]
    boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
    ret = tf.image.crop_and_resize(
        image, boxes, tf.cast(box_ind, tf.int32),
        crop_size=[crop_size, crop_size])
    return ret


def roi_align(featuremap, boxes, resolution):
    """
    Args:
        featuremap: 1xHxWxC
        boxes: [0, 1]
        resolution: output spatial resolution
    Returns:
        NxCx res x res
    """
    # sample 4 locations per roi bin
    ret = crop_and_resize(
        featuremap, boxes,
        tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32),
        resolution * 2)
    ret = tf.nn.avg_pool(ret, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    return ret


class RoiPooling(tf.keras.Model):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size
        self._concat_layer = layers.Concatenate(axis=0)
        self._flatten_layer = layers.Flatten()
        self._max_pool = layers.MaxPooling2D(padding='same')

    def call(self, inputs, training=None, mask=None):
        """
        输入 backbone 的结果和 rpn proposals 的结果(即 RegionProosal 的输出)
        输出 roi pooloing 的结果，即在特征图上，对每个rpn proposal获取一个固定尺寸的特征图
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        # [1, height, width, channels]  [num_rois, 4]
        shared_layers, rois, extractor_stride = inputs
        rois = rois / extractor_stride

        # # TODO: ROI Polling 的细节

        # # 方法一
        # res = []
        # for roi in rois:
        #     ymin = tf.to_int32(roi[0])
        #     xmin = tf.to_int32(roi[1])
        #     ymax = tf.to_int32(roi[2])
        #     xmax = tf.to_int32(roi[3])
        #     res.append(
        #         tf.image.resize_bilinear(shared_layers[:, ymin:ymax + 1, xmin:xmax + 1, :],
        #                                  [self._pool_size, self._pool_size], align_corners=True))
        # net = self._concat_layer(res)
        # net = self._flatten_layer(net)
        # return tf.stop_gradient(net)

        # # 方法二
        # # roi align copy from https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/model_box.py
        # net = roi_align(shared_layers, rois, self._pool_size)
        # net = self._flatten_layer(net)
        # return tf.stop_gradient(net)

        # # 方法三
        # h, w = shared_layers.get_shape().as_list()[1:3]
        # roi_channels = tf.split(rois, 4, axis=1)
        # rois = tf.concat([
        #     roi_channels[0] / tf.to_float(h),
        #     roi_channels[1] / tf.to_float(w),
        #     roi_channels[2] / tf.to_float(h),
        #     roi_channels[3] / tf.to_float(w),
        # ], axis=1)
        # net = tf.image.crop_and_resize(shared_layers,
        #                                rois,
        #                                tf.zeros([tf.shape(rois)[0]], dtype=tf.int32),
        #                                [self._pool_size, self._pool_size])
        # net = self._flatten_layer(net)
        # return tf.stop_gradient(net)

        # 方法四
        # tf-faster-rcnn 中借鉴
        batch_ids = tf.zeros([tf.shape(rois)[0]], dtype=tf.int32)
        h, w = shared_layers.get_shape().as_list()[1:3]
        roi_channels = tf.split(rois, 4, axis=1)
        # bboxes = tf.concat([
        #     roi_channels[1] / tf.to_float(h - 1),
        #     roi_channels[0] / tf.to_float(w - 1),
        #     roi_channels[3] / tf.to_float(h - 1),
        #     roi_channels[2] / tf.to_float(w - 1),
        # ], axis=1)
        bboxes = tf.concat([
            roi_channels[0] / tf.to_float(h),
            roi_channels[1] / tf.to_float(w),
            roi_channels[2] / tf.to_float(h),
            roi_channels[3] / tf.to_float(w),
        ], axis=1)
        pre_pool_size = self._pool_size * 2
        crops = tf.image.crop_and_resize(shared_layers, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                         name="crops")
        return tf.stop_gradient(self._flatten_layer(self._max_pool(crops)))


class RoiHead(tf.keras.Model):
    def __init__(self, num_classes,
                 roi_feature_size=7 * 7 * 512,
                 keep_rate=0.5, weight_decay=0.0005):
        super().__init__()
        self._num_classes = num_classes

        self._fc1 = layers.Dense(4096, name='fc1', activation='relu',
                                 kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                 input_shape=[roi_feature_size]
                                 )
        self._dropout1 = layers.Dropout(rate=1 - keep_rate)

        self._fc2 = layers.Dense(4096, name='fc2', activation='relu',
                                 kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                 )
        self._dropout2 = layers.Dropout(rate=1 - keep_rate)

        self._score_layer = layers.Dense(num_classes, name='roi_head_score', activation=None,
                                         kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                         kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self._roi_bboxes_layer = layers.Dense(4 * num_classes, name='roi_head_bboxes', activation=None,
                                              kernel_initializer=tf.random_normal_initializer(0, 0.001),
                                              kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self._roi_bboxes_reshape_layer = layers.Reshape([num_classes, 4])

        self.build((None, roi_feature_size))
        weights_path = tf.keras.utils.get_file(
            'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
            VGG_16_WEIGHTS_PATH,
            cache_subdir='models',
            file_hash='64373286793e3c8b2b4e3219cbf3544b')
        self.load_weights(weights_path, by_name=True)

    def call(self, inputs, training=None):
        """
        输入 roi pooling 的结果
        对每个 roi pooling 的结果进行预测（预测bboxes）
        :param inputs:  roi_features, [num_rois, len_roi_feature]
        :param training:
        :param mask:
        :return:
        """
        x = self._fc1(inputs)
        x = self._dropout1(x, training)
        x = self._fc2(x)
        x = self._dropout2(x, training)
        score = self._score_layer(x)
        bboxes = self._roi_bboxes_reshape_layer(self._roi_bboxes_layer(x))

        return score, bboxes


class ProposalTarget(tf.keras.Model):
    def __init__(self,
                 pos_iou_threshold=0.5,
                 neg_iou_threshold=0.5,
                 total_num_samples=128,
                 max_pos_samples=32,
                 target_means=None,
                 target_stds=None):
        super().__init__()

        self._pos_iou_threshold = pos_iou_threshold
        self._neg_iou_threshold = neg_iou_threshold
        self._total_num_samples = total_num_samples
        self._max_pos_samples = max_pos_samples

        if target_stds is None:
            target_stds = [1, 1, 1, 1]
        if target_means is None:
            target_means = [0, 0, 0, 0]
        self._target_means = target_means
        self._target_stds = target_stds

    def call(self, inputs, training=None, mask=None):
        """
        不需要训练
        生成训练roi用的数据
        总体过程：
        1. 计算 rois 与 gt_bboxes（即输入数据中的bbox）的iou
        2. 设置与 gt_bboxes 的 max_iou > pos_iou_threshold 的 roi 为正例，设置 max_iou < neg_iou_threshold 的 roi 为反例
        3. 对正例、反例有数量限制，正例数量不大于 max_pos_samples，正例反例总数不超过 max_pos_samples
        4. 最终输出三个结果：
                1）参与训练的 roi 的编号
                2）每个参与训练的 roi 的label [0, num_classes)，可直接用于 cls loss
                3）pos rois 对应的 gt，可直接用于 reg loss
                4）pos anchors num，scalar
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        rois, gt_bboxes, gt_labels, image_shape = inputs

        # [rois_size, gt_bboxes_size]
        labels = -tf.ones((rois.shape[0],), tf.int32)
        iou = pairwise_iou(rois, gt_bboxes)

        # 设置与任意gt_bbox的iou > pos_iou_threshold 的 roi 为正例
        # 设置与所有gt_bbox的iou < neg_iou_threshold 的 roi 为反例
        max_ious = tf.reduce_max(iou, axis=1)
        gt_bbox_idx = tf.argmax(iou, axis=1)
        labels = tf.where(max_ious >= self._pos_iou_threshold, tf.ones_like(labels), labels)
        labels = tf.where(tf.logical_and(max_ious >= self._neg_iou_threshold,
                                         max_ious < self._pos_iou_threshold), tf.zeros_like(labels), labels)

        # 筛选正例和反例
        pos_index = tf.where(tf.equal(labels, 1))[:, 0]
        neg_index = tf.where(tf.equal(labels, 0))[:, 0]
        total_pos_num = tf.size(pos_index)  # 计算正例真实数量
        total_neg_num = tf.size(neg_index)  # 计算反例真实数量
        cur_pos_num = tf.minimum(total_pos_num, self._max_pos_samples)  # 根据要求，修正正例数量
        cur_neg_num = tf.minimum(self._total_num_samples - cur_pos_num, total_neg_num)  # 根据要求，修正反例数量
        pos_index = tf.random_shuffle(pos_index)[:cur_pos_num]  # 随机获取正例
        neg_index = tf.random_shuffle(neg_index)[:cur_neg_num]  # 随机获取反例
        tf_logging.debug('roi training has %d pos samples and %d neg samples' % (cur_pos_num, cur_neg_num))

        # 生成最终结果
        roi_training_idx = tf.concat([pos_index, neg_index], axis=0)
        roi_cls_gt_labels = tf.multiply(tf.gather(gt_labels, tf.gather(gt_bbox_idx, roi_training_idx)),
                                        tf.gather(labels, roi_training_idx))
        roi_reg_gt_txtytwth = encode_bbox_with_mean_and_std(tf.gather(rois, pos_index),
                                                            tf.gather(gt_bboxes, tf.gather(gt_bbox_idx, pos_index)),
                                                            self._target_means, self._target_stds)

        return tf.stop_gradient(roi_training_idx), tf.stop_gradient(roi_cls_gt_labels), \
               tf.stop_gradient(roi_reg_gt_txtytwth), tf.stop_gradient(cur_pos_num)
