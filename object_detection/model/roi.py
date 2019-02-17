import tensorflow as tf
from object_detection.utils.bbox_tf import pairwise_iou
from tensorflow.python.platform import tf_logging

layers = tf.keras.layers

__all__ = ['RoiHead', 'RoiPooling', 'RoiTrainingProposal']


def crop_and_resize(image, boxes, box_ind, crop_size, pad_border=True):
    assert isinstance(crop_size, int), crop_size
    boxes = tf.stop_gradient(boxes)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
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
    ret = tf.transpose(ret, [0, 3, 1, 2])  # ncss
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

    def call(self, inputs, training=None, mask=None):
        """
        输入 backbone 的结果和 rpn proposals 的结果(即 RPNProposal 的输出)
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
        #                                  [self._pool_size, self._pool_size]))
        # net = self._concat_layer(res)
        # net = self._flatten_layer(net)
        # return tf.stop_gradient(net)

        # 方法二
        # roi align copy from https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/model_box.py
        h, w = shared_layers.get_shape().as_list()[1:3]
        roi_channels = tf.split(rois, 4, axis=1)
        rois = tf.concat([
            roi_channels[0] / tf.to_float(h),
            roi_channels[1] / tf.to_float(w),
            roi_channels[2] / tf.to_float(h),
            roi_channels[3] / tf.to_float(w),
        ], axis=1)

        net = self._flatten_layer(roi_align(shared_layers, rois, self._pool_size))
        return tf.stop_gradient(net)


class RoiHead(tf.keras.Model):
    def __init__(self, num_classes,
                 fc1=None, fc2=None,
                 keep_rate=0.5, weight_decay=0.0005):
        super().__init__()
        self._num_classes = num_classes

        if fc1 is None:
            self._fc1 = layers.Dense(1024, name='fc1',
                                     kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        else:
            self._fc1 = fc1
        if fc2 is None:
            self._fc2 = layers.Dense(1024, name='fc2',
                                     kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        else:
            self._fc2 = fc2
        self._dropout1 = layers.Dropout(rate=1 - keep_rate)
        self._dropout2 = layers.Dropout(rate=1 - keep_rate)

        self._score_prediction = layers.Dense(num_classes, name='roi_head_score',
                                              kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                              kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self._bbox_prediction = layers.Dense(4 * num_classes, name='roi_head_bboxes',
                                             kernel_initializer=tf.random_normal_initializer(0, 0.001),
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))

    def call(self, inputs, training=None, mask=None):
        """
        输入 roi pooling 的结果
        对每个 roi pooling 的结果进行预测（预测bboxes）
        :param inputs:  roi_features, [num_rois, len_roi_feature]
        :param training:
        :param mask:
        :return:
        """
        net = self._fc1(inputs)
        net = self._dropout1(net, training)
        net = self._fc2(net)
        net = self._dropout2(net, training)
        roi_score = tf.reshape(self._score_prediction(net), [-1, self._num_classes])
        roi_bboxes_txtytwth = tf.reshape(self._bbox_prediction(net), [-1, self._num_classes, 4])

        return roi_score, roi_bboxes_txtytwth


class RoiTrainingProposal(tf.keras.Model):
    def __init__(self,
                 pos_iou_threshold=0.5,
                 neg_iou_threshold=0.5,
                 total_num_samples=128,
                 max_pos_samples=32, ):
        super().__init__()

        self._pos_iou_threshold = pos_iou_threshold
        self._neg_iou_threshold = neg_iou_threshold
        self._total_num_samples = total_num_samples
        self._max_pos_samples = max_pos_samples

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
                2）每个参与训练的 roi 的label（正例还是反例）
                3）每个参与训练的 roi 对应的gt_bboxes编号（即与每个 roi 的iou最大的gt_bboxes编号）
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        rois, gt_bboxes, image_shape = inputs

        # [rois_size, gt_bboxes_size]
        labels = -tf.ones((rois.shape[0],), tf.int32)
        iou = pairwise_iou(rois, gt_bboxes)

        # 设置与任意gt_bbox的iou > pos_iou_threshold 的 roi 为正例
        # 设置与所有gt_bbox的iou < neg_iou_threshold 的 roi 为反例
        max_scores = tf.reduce_max(iou, axis=1)
        gt_bbox_idx = tf.argmax(iou, axis=1)
        labels = tf.where(max_scores >= self._pos_iou_threshold, tf.ones_like(labels), labels)
        labels = tf.where(max_scores < self._neg_iou_threshold, tf.zeros_like(labels), labels)

        # 计算正反例真实数量
        total_pos_num = tf.reduce_sum(tf.where(tf.equal(labels, 1), tf.ones_like(labels), tf.zeros_like(labels)))
        total_neg_num = tf.reduce_sum(tf.where(tf.equal(labels, 0), tf.ones_like(labels), tf.zeros_like(labels)))

        # 根据要求，修正正反例数量
        cur_pos_num = tf.minimum(total_pos_num, self._max_pos_samples)
        cur_neg_num = tf.minimum(self._total_num_samples - cur_pos_num, total_neg_num)
        # tf_logging.info('roi training has %d pos samples and %d neg samples' % (cur_pos_num, cur_neg_num))

        # 随机选择正例和反例
        total_pos_index = tf.squeeze(tf.where(tf.equal(labels, 1)), axis=1)
        total_neg_index = tf.squeeze(tf.where(tf.equal(labels, 0)), axis=1)
        pos_index = tf.gather(total_pos_index, tf.random_shuffle(tf.range(0, total_pos_num))[:cur_pos_num])
        neg_index = tf.gather(total_neg_index, tf.random_shuffle(tf.range(0, total_neg_num))[:cur_neg_num])

        # 生成最终结果
        selected_idx = tf.concat([pos_index, neg_index], axis=0)
        return tf.stop_gradient(selected_idx), \
               tf.stop_gradient(tf.gather(labels, selected_idx)), \
               tf.stop_gradient(tf.gather(gt_bbox_idx, selected_idx)), \
               tf.stop_gradient(cur_pos_num)
