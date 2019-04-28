import tensorflow as tf
from object_detection.utils.bbox_transform import decode_bbox_with_mean_and_std
from object_detection.utils.bbox_tf import bboxes_clip_filter
from tensorflow.python.platform import tf_logging

layers = tf.keras.layers

__all__ = ['RegionProposal']


class RegionProposal(tf.keras.Model):
    def __init__(self,
                 num_anchors=9,
                 num_pre_nms_train=12000,
                 num_post_nms_train=2000,
                 num_pre_nms_test=6000,
                 num_post_nms_test=300,
                 nms_iou_threshold=0.7,
                 target_means=None,
                 target_stds=None):
        super().__init__()

        self._num_anchors = num_anchors
        self._num_pre_nms_train = num_pre_nms_train
        self._num_post_nms_train = num_post_nms_train
        self._num_pre_nms_test = num_pre_nms_test
        self._num_post_nms_test = num_post_nms_test
        self._nms_iou_threshold = nms_iou_threshold

        if target_stds is None:
            target_stds = [1, 1, 1, 1]
        if target_means is None:
            target_means = [0, 0, 0, 0]
        self._target_means = target_means
        self._target_stds = target_stds

    def call(self, inputs, training=None, mask=None):
        """
        生成 rpn 的结果，即一组 bboxes，用于后续 roi pooling
        总体过程：
        1. 使用anchors和rpn_pred修正，获取所有预测结果。
        2. 对选中修正后的anchors进行处理（剪裁）。
        3. 根据rpn_score获取num_pre_nms个anchors。
        4. 进行nms。
        5. 根据rpn_score排序，获取num_post_nms个anchors作为proposal结果。
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        # bboxes_txtytwth shape: [num_anchors*feature_width*feature_height, 4]
        # anchors shape: [num_anchors*feature_width*feature_height, 4]
        # scores shape: [feature_width*feature_height*num_anchors,]
        # image_shape shape: [2, ]
        bboxes_txtytwth, anchors, scores, image_shape = inputs

        # 1. 使用anchors使用rpn_pred修正，获取所有预测结果。
        # [num_anchors*feature_width*feature_height, 4]
        decoded_bboxes = decode_bbox_with_mean_and_std(anchors, bboxes_txtytwth,
                                                       self._target_means, self._target_stds)

        # 2. 对选中修正后的anchors进行处理
        decoded_bboxes, _ = bboxes_clip_filter(decoded_bboxes, 0, image_shape[0], image_shape[1])

        # # 3. 根据rpn_score获取num_pre_nms个anchors。
        # num_pre_nms = self._num_pre_nms_train if training else self._num_pre_nms_test
        # cur_top_k = tf.minimum(num_pre_nms, tf.size(scores))
        # scores, selected_idx = tf.nn.top_k(scores, k=cur_top_k, sorted=False)
        # decoded_bboxes = tf.gather(decoded_bboxes, selected_idx)

        # 4. 进行nms。
        # 5. 根据rpn_score排序，获取num_post_nms个anchors作为proposal结果。
        num_post_nms = self._num_post_nms_train if training else self._num_post_nms_test
        selected_idx = tf.image.non_max_suppression(tf.to_float(decoded_bboxes), scores,
                                                    max_output_size=num_post_nms,
                                                    iou_threshold=self._nms_iou_threshold)

        tf_logging.debug('rpn proposal net generate %d proposals' % tf.size(selected_idx))

        # 不参与训练，所以需要设置 tf.stop_gradient
        return tf.stop_gradient(tf.gather(decoded_bboxes, selected_idx))
