import tensorflow as tf

layers = tf.keras.layers


class RoiPooling(tf.keras.Model):
    def __init__(self, num_rois, pool_size):
        super().__init__()
        self._num_rois = num_rois
        self._pool_size = pool_size
        self._concat_layer = layers.Concatenate(axis=0)

    def call(self, inputs, training=None, mask=None):
        """
        inputs include shapred_layers [1, height, width, channels] and rois [1, num_rois, 4]
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        shared_layers = inputs[0]
        rois = inputs[1]
        res = []
        for idx in range(self._num_rois):
            x = tf.to_int32(rois[0, idx, 0])
            y = tf.to_int32(rois[0, idx, 1])
            w = tf.to_int32(rois[0, idx, 2])
            h = tf.to_int32(rois[0, idx, 3])
            res.append(tf.image.resize_bilinear(shared_layers[:, y:y+h, x:x+w, :], [self._pool_size, self._pool_size]))
        return self._concat_layer(res, axis=0)
