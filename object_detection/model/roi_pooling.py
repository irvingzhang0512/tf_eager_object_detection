import tensorflow as tf

layers = tf.keras.layers

__all__ = ['RoiPoolingCropAndResize', 'RoiPoolingRoiAlign']


class RoiPoolingCropAndResize(tf.keras.Model):
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

        batch_ids = tf.zeros([tf.shape(rois)[0]], dtype=tf.int32)
        h, w = shared_layers.get_shape().as_list()[1:3]
        roi_channels = tf.split(rois, 4, axis=1)
        bboxes = tf.concat([
            roi_channels[1] / tf.to_float(h - 1),
            roi_channels[0] / tf.to_float(w - 1),
            roi_channels[3] / tf.to_float(h - 1),
            roi_channels[2] / tf.to_float(w - 1),
        ], axis=1)
        pre_pool_size = self._pool_size * 2

        # 重大bug…… shared_layers 还是需要参与反向传播的……，bboxes不参加
        crops = tf.image.crop_and_resize(shared_layers,
                                         tf.stop_gradient(bboxes),
                                         box_ind=tf.to_int32(batch_ids),
                                         crop_size=[pre_pool_size, pre_pool_size],
                                         name="crops")
        return self._flatten_layer(self._max_pool(crops))


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


class RoiPoolingRoiAlign(tf.keras.Model):
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

        net = roi_align(shared_layers, tf.stop_gradient(rois), self._pool_size)
        net = self._flatten_layer(net)
        return net
