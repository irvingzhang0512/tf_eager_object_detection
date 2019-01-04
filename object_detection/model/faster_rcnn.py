import tensorflow as tf
from object_detection.model.rpn import RPNHead, RPNProposal, RPNTrainingProposal
from object_detection.model.roi_pooling import RoiPooling
from object_detection.utils.anchors import generate_anchors_np

layers = tf.keras.layers


class BaseFasterRcnn(tf.keras.Model):
    def __init__(self,
                 image_size,
                 extractor_stride,
                 extractor,
                 ratios,
                 scales,
                 roi_pool_size,
                 num_rois=256,
                 ):
        super().__init__()
        self._extractor = extractor
        self._ratios = ratios
        self._scales = scales
        self._roi_pool_size = roi_pool_size
        self._num_rois = num_rois

        self._anchors = generate_anchors_np(scales, ratios, image_size, extractor_stride)

        self._rpn_head = RPNHead(len(ratios) * len(scales))
        self._rpn_training_proposal = RPNTrainingProposal()
        self._rpn_proposal = RPNProposal()
        self._roi_pool_layer = RoiPooling(num_rois=num_rois, pool_size=roi_pool_size)

    def call(self, inputs, training=None, mask=None):
        """
        build faster r-cnn model
        :param inputs:          shape [1, height, width, 3], [num_bboxes, 4]
        :param training:
        :param mask:
        :return:
        """
        # [1, height, width, 3], [num_bboxes, 4]
        image, gt_bboxes = inputs

        # extract features
        # shape like [1, height/32, width/32, 512 or 1024]
        shared_features = self._extractor(image, training, mask)

        # rpn head
        # shape [num_anchors*feature_width*feature_height, 2], [num_anchors*feature_width*feature_height, 4]
        rpn_score, rpn_bbox = self._rpn_head(shared_features, training, mask)

        rpn_training_idx, rpn_training_labels = self._rpn_training_proposal(self._anchors, gt_bboxes)
        # TODO: cal rpn cls loss by rpn_training_idx, rpn_training_labels & rpn_score
        # 计算分类损失函数：正反例都有了，使用softmax直接计算就行（label可能还要处理一下）
        # TODO: cal rpn bbox reg loss by rpn_training_idx, rpn_training_labels & gt_bboxes
        # 计算边框回归损失函数，只计算正例的损失函数，要知道每个正例anchor对应的gt_bboxes是哪个，计算tx ty tw th
        # 计算结果与预测结果（即rpn_bbox）计算smooth L1损失函数

        # roi pooling
        rpn_proposals = self._rpn_proposal(rpn_bbox, self._anchors, rpn_score[:, 1])
        roi_features = self._roi_pool_layer(shared_features, rpn_proposals)


