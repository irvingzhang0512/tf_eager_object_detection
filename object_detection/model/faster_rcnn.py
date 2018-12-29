import tensorflow as tf
from object_detection.model.rpn import RPN
from object_detection.model.roi_pooling import RoiPooling

layers = tf.keras.layers


class BaseFasterRcnn(tf.keras.Model):
    def __init__(self,
                 extractor,
                 ratios,
                 scales,
                 roi_pool_size,
                 num_rois,
                 ):
        super().__init__()
        self._extractor = extractor
        self._ratios = ratios
        self._scales = scales
        self._roi_pool_size = roi_pool_size
        self._num_rois = num_rois

        self._rpn_layer = RPN(len(ratios) * len(scales))
        self._roi_pool_layer = RoiPooling(num_rois=num_rois, pool_size=roi_pool_size)

    def call(self, inputs, training=None, mask=None):
        """
        build faster r-cnn model
        :param inputs:          shape [1, height, width, 3]
        :param training:
        :param mask:
        :return:
        """
        # extract features
        # shape like [1, height/32, width/32, 512 or 1024]
        shared_features = self._extractor(inputs, training, mask)

        # rpn
        # shape [1, -1, 2], [1, -1, 4]
        rpn_score, rpn_bbox = self._rpn_layer(shared_features, training, mask)

        # roi pooling
        raw_rois = self._roi_pool_layer(shared_features, rpn_bbox)

