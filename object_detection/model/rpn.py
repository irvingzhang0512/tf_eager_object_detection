import tensorflow as tf

layers = tf.keras.layers


class RPN(tf.keras.Model):
    def __init__(self, num_anchors):
        super().__init__()
        self._rpn_conv = layers.Conv2D(512, [3, 3], activation=tf.nn.relu,
                                       padding='same', name='rpn_first_conv')

        self._score_num = num_anchors * 2
        self._rpn_score_conv = layers.Conv2D(self._score_num, [1, 1], activation=tf.nn.sigmoid,
                                             padding='same', name='rpn_score_conv')
        self._rpn_score_reshape_layer = layers.Reshape([1, -1, 2])

        self._bbox_num = num_anchors * 4
        self._rpn_bbox_conv = layers.Conv2D(self._bbox_num, [1, 1], padding='same', name='rpn_bbox_conv')
        self._rpn_bbox_reshape_layer = layers.Reshape([1, -1, 4])

    def call(self, inputs, training=None, mask=None):
        x = self._rpn_conv(inputs)
        rpn_score = self._rpn_score_conv(x)
        rpn_score_reshape = self._rpn_score_reshape_layer(rpn_score)
        rpn_bbox = self._rpn_bbox_conv(x)
        rpn_bbox_reshape = self._rpn_bbox_reshape_layer(rpn_bbox)
        return rpn_score_reshape, rpn_bbox_reshape
