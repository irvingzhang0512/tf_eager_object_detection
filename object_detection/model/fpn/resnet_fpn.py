import tensorflow as tf
import numpy as np
from object_detection.model.fpn.base_fpn_model import BaseFPN
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block

layers = tf.keras.layers

BASE_WEIGHTS_PATH = (
    'https://github.com/keras-team/keras-applications/releases/download/resnet/')
WEIGHTS_HASHES = {
    'resnet50': ('2cb95161c43110f7111970584f804107',
                 '4d473c1dd8becc155b73f8504c6f6626'),
    'resnet101': ('f1aeb4b969a6efcfb50fad2f0c20cfc5',
                  '88cf7a10940856eca736dc7b7e228a21'),
    'resnet152': ('100835be76be38e30d865e96f2aaae62',
                  'ee4c566cf9a93f14d82f913c2dc6dd0c'),
    'resnet50v2': ('3ef43a0b657b3be2300d5770ece849e0',
                   'fac2f116257151a9d068a22e544a4917'),
    'resnet101v2': ('6343647c601c52e1368623803854d971',
                    'c0ed64b8031c3730f411d2eb4eea35b5'),
    'resnet152v2': ('a49b44d1979771252814e80f8ec446f9',
                    'ed17cf2e0169df9d443503ef94b23b33'),
    'resnext50': ('67a5b30d522ed92f75a1f16eef299d1a',
                  '62527c363bdd9ec598bed41947b379fc'),
    'resnext101': ('34fb605428fcc7aa4d62f44404c11509',
                   '0f678c91647380debd923963594981b3')
}


def block1(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None,
           trainable=True, weight_decay=0.0001):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3
    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                 name=name + '_0_conv', trainable=trainable,
                                 kernel_initializer='he_normal')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn', trainable=False)(shortcut, training=False)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv', trainable=trainable,
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn', trainable=False)(x, training=False)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME',
                      name=name + '_2_conv', trainable=trainable,
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn', trainable=False)(x, training=False)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv', trainable=trainable,
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn', trainable=False)(x, training=False)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None, trainable=True, weight_decay=0.0001):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + '_block1',
               trainable=trainable, weight_decay=weight_decay)
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i),
                   trainable=trainable, weight_decay=weight_decay)
    return x


def get_resnet_model(stack_fn, model_name='resnet', weight_decay=0.0001):
    img_input = layers.Input(shape=(None, None, 3))

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv', trainable=True, padding='valid',
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                      kernel_initializer='he_normal')(x)

    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5,
                                  name='conv1_bn', trainable=False)(x, training=False)
    x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    c2, c3, c4, c5 = stack_fn(x)

    model = tf.keras.Model(img_input, [c2, c3, c4, c5], name=model_name)

    # Load weights.
    if model_name in WEIGHTS_HASHES:
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
        file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = tf.keras.utils.get_file(file_name,
                                               BASE_WEIGHTS_PATH + file_name,
                                               cache_subdir='models',
                                               file_hash=file_hash)
        model.load_weights(weights_path, by_name=True)
        tf.logging.info('successfully load keras pre-trained weights for {} extractor'.format(model_name))

    return model


def resnet_arg_scope(
        is_training=True, weight_decay=0.0001, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


def resnet_base(scope_name, is_training=True):
    img_batch = layers.InputLayer(input_shape=(None, None, 3))

    if scope_name == 'resnet_v1_50':
        middle_num_units = 6
    elif scope_name == 'resnet_v1_101':
        middle_num_units = 23
    else:
        raise NotImplementedError('We only support resnet_v1_50 or resnet_v1_101. Check your network name....yjr')

    blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
              resnet_v1_block('block3', base_depth=256, num_units=middle_num_units, stride=2),
              resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with tf.variable_scope(scope_name, scope_name):
            # Do the first few layers manually, because 'SAME' padding can behave inconsistently
            # for images of different sizes: sometimes 0, sometimes 1
            net = resnet_utils.conv2d_same(
                img_batch, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')

    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        c2, end_points_c2 = resnet_v1.resnet_v1(net,
                                                blocks[0:1],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        c3, end_points_c3 = resnet_v1.resnet_v1(c2,
                                                blocks[1:2],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        c4, end_points_c4 = resnet_v1.resnet_v1(c3,
                                                blocks[2:3],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        _, end_points_c5 = resnet_v1.resnet_v1(c4,
                                               blocks[3:4],
                                               global_pool=False,
                                               include_root_block=False,
                                               scope=scope_name)

    feature_dict = {'C2': end_points_c2['{}/block1/unit_2/bottleneck_v1'.format(scope_name)],
                    'C3': end_points_c3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)],
                    'C4': end_points_c4['{}/block3/unit_{}/bottleneck_v1'.format(scope_name, middle_num_units - 1)],
                    'C5': end_points_c5['{}/block4/unit_3/bottleneck_v1'.format(scope_name)], }

    extractor = tf.keras.Model(img_batch,
                               [feature_dict['C2'], feature_dict['C3'], feature_dict['C4'], feature_dict['C5']])
    return extractor


def get_resnet_v1_extractor(depth, weight_decay=0.0001):
    if depth == 50:
        def stack_fn(x):
            c2 = stack1(x, 64, 3, stride1=1, name='conv2', trainable=True, weight_decay=weight_decay)
            c3 = stack1(c2, 128, 4, name='conv3', weight_decay=weight_decay)
            c4 = stack1(c3, 256, 6, name='conv4', weight_decay=weight_decay)
            c5 = stack1(c4, 512, 3, name='conv5', weight_decay=weight_decay)
            return c2, c3, c4, c5
    elif depth == 101:
        def stack_fn(x):
            c2 = stack1(x, 64, 3, stride1=1, name='conv2', trainable=True, weight_decay=weight_decay)
            c3 = stack1(c2, 128, 4, name='conv3', weight_decay=weight_decay)
            c4 = stack1(c3, 256, 23, name='conv4', weight_decay=weight_decay)
            c5 = stack1(c4, 512, 3, name='conv5', weight_decay=weight_decay)
            return c2, c3, c4, c5
    elif depth == 152:
        def stack_fn(x):
            c2 = stack1(x, 64, 3, stride1=1, name='conv2', trainable=True, weight_decay=weight_decay)
            c3 = stack1(c2, 128, 8, name='conv3', weight_decay=weight_decay)
            c4 = stack1(c3, 256, 36, name='conv4', weight_decay=weight_decay)
            c5 = stack1(c4, 512, 3, name='conv5', weight_decay=weight_decay)
            return c2, c3, c4, c5
    else:
        raise ValueError('unknown depth {}'.format(depth))

    return get_resnet_model(stack_fn,
                            model_name='resnet{}'.format(depth),
                            weight_decay=weight_decay)


class ResnetRoiHead(tf.keras.Model):
    def __init__(self, num_classes,
                 roi_feature_size=(7, 7, 256),
                 keep_rate=0.5, weight_decay=0.0001, ):
        super().__init__()
        self._num_classes = num_classes

        self._fc1 = layers.Dense(1024, name='fc1', activation='relu',
                                 kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                 input_shape=[roi_feature_size]
                                 )
        self._dropout1 = layers.Dropout(rate=1 - keep_rate)

        self._fc2 = layers.Dense(1024, name='fc2', activation='relu',
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
        self._flatten_layer = layers.Flatten()

    def call(self, inputs, training=None):
        """
        输入 roi pooling 的结果
        对每个 roi pooling 的结果进行预测（预测bboxes）
        :param inputs:  roi_features, [num_rois, pool_size, pool_size, num_channels]
        :param training:
        :return:
        """
        x = self._flatten_layer(inputs)
        x = self._fc1(x)
        # x = self._dropout1(x, training)
        x = self._fc2(x)
        # x = self._dropout2(x, training)
        score = self._score_layer(x)
        bboxes = self._roi_bboxes_layer(x)

        return score, bboxes


class ResnetFpnNeck(tf.keras.Model):
    def __init__(self, top_down_dims=256, weight_decay=0.0001, use_bias=True):
        super().__init__()

        self._build_p5_conv = layers.Conv2D(top_down_dims, 1, strides=1, use_bias=use_bias, name='build_p5',
                                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                            kernel_initializer='he_normal')
        self._build_p6_max_pooling = layers.MaxPooling2D(strides=2, pool_size=(1, 1), name='build_p6')

        self._build_p4_reduce_dims = layers.Conv2D(top_down_dims, 1, strides=1, use_bias=use_bias,
                                                   name='build_p4_reduce_dims',
                                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                                   kernel_initializer='he_normal')
        self._build_p4_fusion = layers.Add(name='build_p4_fusion')
        self._build_p4 = layers.Conv2D(top_down_dims, 3, 1, use_bias=use_bias,
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       padding='same', name='build_p4',
                                       kernel_initializer='he_normal')

        self._build_p3_reduce_dims = layers.Conv2D(top_down_dims, 1, strides=1, use_bias=use_bias,
                                                   name='build_p3_reduce_dims',
                                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                                   kernel_initializer='he_normal')
        self._build_p3_fusion = layers.Add(name='build_p3_fusion')
        self._build_p3 = layers.Conv2D(top_down_dims, 3, 1, use_bias=use_bias,
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       padding='same', name='build_p3',
                                       kernel_initializer='he_normal')

        self._build_p2_reduce_dims = layers.Conv2D(top_down_dims, 1, strides=1, use_bias=use_bias,
                                                   name='build_p2_reduce_dims',
                                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                                   kernel_initializer='he_normal')
        self._build_p2_fusion = layers.Add(name='build_p2_fusion')
        self._build_p2 = layers.Conv2D(top_down_dims, 3, 1, use_bias=use_bias,
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       padding='same', name='build_p2',
                                       kernel_initializer='he_normal')

    def call(self, inputs, training=None, mask=None):
        c2, c3, c4, c5 = inputs

        # build p5 & p6
        p5 = self._build_p5_conv(c5)
        p6 = self._build_p6_max_pooling(p5)

        # build p4
        h, w = tf.shape(c4)[1], tf.shape(c4)[2]
        upsample_p5 = tf.image.resize_bilinear(p5, (h, w), name='build_p4_resize')
        reduce_dims_c4 = self._build_p4_reduce_dims(c4)
        p4 = self._build_p4_fusion([upsample_p5 * 0.5, reduce_dims_c4 * 0.5])

        # build p3
        h, w = tf.shape(c3)[1], tf.shape(c3)[2]
        upsample_p4 = tf.image.resize_bilinear(p4, (h, w), name='build_p3_resize')
        reduce_dims_c3 = self._build_p3_reduce_dims(c3)
        p3 = self._build_p3_fusion([upsample_p4 * 0.5, reduce_dims_c3 * 0.5])

        # build p2
        h, w = tf.shape(c2)[1], tf.shape(c2)[2]
        upsample_p3 = tf.image.resize_bilinear(p3, (h, w), name='build_p2_resize')
        reduce_dims_c2 = self._build_p2_reduce_dims(c2)
        p2 = self._build_p2_fusion([upsample_p3 * 0.5, reduce_dims_c2 * 0.5])

        p4 = self._build_p4(p4)
        p3 = self._build_p3(p3)
        p2 = self._build_p2(p2)

        return p2, p3, p4, p5, p6


class ResnetV1Fpn(BaseFPN):
    def __init__(self,
                 depth=50,
                 roi_head_keep_dropout_rate=0.5,

                 # 通用参数
                 roi_feature_size=(7, 7, 256),
                 num_classes=21,
                 weight_decay=0.0001,

                 # fpn 特有参数
                 level_name_list=('p2', 'p3', 'p4', 'p5', 'p6'),
                 min_level=2,
                 max_level=5,
                 top_down_dims=256,

                 # fpn 中 anchors 特有参数
                 anchor_stride_list=(4, 8, 16, 32, 64),
                 base_anchor_size_list=(32, 64, 128, 256, 512),
                 ratios=(0.5, 1.0, 2.0),
                 scales=(1.,),

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
                 roi_pooling_max_pooling_flag=True,

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
                 prediction_score_threshold=0.3,
                 ):
        self._depth = depth
        self._roi_head_keep_dropout_rate = roi_head_keep_dropout_rate
        self._top_down_dims = top_down_dims
        super().__init__(
            roi_feature_size=roi_feature_size,
            num_classes=num_classes,
            weight_decay=weight_decay,

            # fpn 特有参数
            level_name_list=level_name_list,
            min_level=min_level,
            max_level=max_level,

            # fpn 中 anchors 特有参数
            anchor_stride_list=anchor_stride_list,
            base_anchor_size_list=base_anchor_size_list,
            ratios=ratios,
            scales=scales,

            # region proposal & anchor target 通用参数
            rpn_proposal_means=rpn_proposal_means,
            rpn_proposal_stds=rpn_proposal_stds,

            # region proposal 参数
            rpn_proposal_num_pre_nms_train=rpn_proposal_num_pre_nms_train,
            rpn_proposal_num_post_nms_train=rpn_proposal_num_post_nms_train,
            rpn_proposal_num_pre_nms_test=rpn_proposal_num_pre_nms_test,
            rpn_proposal_num_post_nms_test=rpn_proposal_num_post_nms_test,
            rpn_proposal_nms_iou_threshold=rpn_proposal_nms_iou_threshold,

            # anchor target 以及相关损失函数参数
            rpn_sigma=rpn_sigma,
            rpn_training_pos_iou_threshold=rpn_training_pos_iou_threshold,
            rpn_training_neg_iou_threshold=rpn_training_neg_iou_threshold,
            rpn_training_total_num_samples=rpn_training_total_num_samples,
            rpn_training_max_pos_samples=rpn_training_max_pos_samples,

            # roi head & proposal target 参数
            roi_proposal_means=roi_proposal_means,
            roi_proposal_stds=roi_proposal_stds,

            # roi pooling 参数
            roi_pool_size=roi_pool_size,
            roi_pooling_max_pooling_flag=roi_pooling_max_pooling_flag,

            # proposal target 以及相关损失函数参数
            roi_sigma=roi_sigma,
            roi_training_pos_iou_threshold=roi_training_pos_iou_threshold,
            roi_training_neg_iou_threshold=roi_training_neg_iou_threshold,
            roi_training_total_num_samples=roi_training_total_num_samples,
            roi_training_max_pos_samples=roi_training_max_pos_samples,

            # prediction 参数
            prediction_max_objects_per_image=prediction_max_objects_per_image,
            prediction_max_objects_per_class=prediction_max_objects_per_class,
            prediction_nms_iou_threshold=prediction_nms_iou_threshold,
            prediction_score_threshold=prediction_score_threshold,
        )

    def _get_roi_head(self):
        return ResnetRoiHead(num_classes=self.num_classes,
                             roi_feature_size=self.roi_feature_size,
                             keep_rate=self._roi_head_keep_dropout_rate,
                             weight_decay=self.weight_decay,
                             )

    def _get_extractor(self):
        return get_resnet_v1_extractor(self._depth, weight_decay=self.weight_decay)

    def _get_neck(self):
        return ResnetFpnNeck(top_down_dims=self._top_down_dims, weight_decay=self.weight_decay)

    def load_fpn_tensorflow_resnet50_weights(self, ckpt_file_path):
        reader = tf.train.load_checkpoint(ckpt_file_path)

        extractor = self.get_layer('resnet50')
        extractor_dict = {
            "resnet_v1_50/conv1/": "conv1_conv",
            "resnet_v1_50/conv1/BatchNorm/": "conv1_bn",

            "resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/": "conv5_block1_0_conv",
            "resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/": "conv5_block1_0_bn",
            "resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/": "conv5_block1_1_conv",
            "resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/": "conv5_block1_1_bn",
            "resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/": "conv5_block1_2_conv",
            "resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/": "conv5_block1_2_bn",
            "resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/": "conv5_block1_3_conv",
            "resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/": "conv5_block1_3_bn",

            "resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/": "conv5_block2_1_conv",
            "resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/": "conv5_block2_1_bn",
            "resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/": "conv5_block2_2_conv",
            "resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/": "conv5_block2_2_bn",
            "resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/": "conv5_block2_3_conv",
            "resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/": "conv5_block2_3_bn",

            "resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/": "conv5_block3_1_conv",
            "resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/": "conv5_block3_1_bn",
            "resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/": "conv5_block3_2_conv",
            "resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/": "conv5_block3_2_bn",
            "resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/": "conv5_block3_3_conv",
            "resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/": "conv5_block3_3_bn",
        }

        keras_format = '{}_{}_{}_{}'  # conv5_block1_0_bn
        ckpt_format = 'resnet_v1_50/{}/{}/bottleneck_v1/{}/{}'  # resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/

        # block1 - unit_1 - shortcut
        # conv2 - block1 - 0
        extractor_dict[ckpt_format.format('block1', 'unit_1', 'shortcut', 'BatchNorm/')] = keras_format.format('conv2',
                                                                                                               'block1',
                                                                                                               0, 'bn')
        extractor_dict[ckpt_format.format('block1', 'unit_1', 'shortcut', '')] = keras_format.format('conv2', 'block1',
                                                                                                     0, 'conv')
        # block1 - unit_1-3 - conv1-3
        # conv2 - block1-3 - 1-3
        for i in range(1, 4):
            for j in range(1, 4):
                key = ckpt_format.format('block1', 'unit_%d' % i, 'conv%d' % j, '')
                value = keras_format.format('conv2', 'block%d' % i, j, 'conv')
                extractor_dict[key] = value
                key = ckpt_format.format('block1', 'unit_%d' % i, 'conv%d' % j, 'BatchNorm/')
                value = keras_format.format('conv2', 'block%d' % i, j, 'bn')
                extractor_dict[key] = value

        # block2 - unit_1 - shortcut
        # conv3 block1 0
        extractor_dict[ckpt_format.format('block2', 'unit_1', 'shortcut', 'BatchNorm/')] = keras_format.format('conv3',
                                                                                                               'block1',
                                                                                                               0, 'bn')
        extractor_dict[ckpt_format.format('block2', 'unit_1', 'shortcut', '')] = keras_format.format('conv3', 'block1',
                                                                                                     0, 'conv')
        # block2 unit_1-4 conv1-3
        # conv3 block1-4 1-3
        for i in range(1, 5):
            for j in range(1, 4):
                key = ckpt_format.format('block2', 'unit_%d' % i, 'conv%d' % j, '')
                value = keras_format.format('conv3', 'block%d' % i, j, 'conv')
                extractor_dict[key] = value
                key = ckpt_format.format('block2', 'unit_%d' % i, 'conv%d' % j, 'BatchNorm/')
                value = keras_format.format('conv3', 'block%d' % i, j, 'bn')
                extractor_dict[key] = value

        # block3 - unit_1 - shortcut
        # conv4 block1 0
        extractor_dict[ckpt_format.format('block3', 'unit_1', 'shortcut', 'BatchNorm/')] = keras_format.format('conv4',
                                                                                                               'block1',
                                                                                                               0, 'bn')
        extractor_dict[ckpt_format.format('block3', 'unit_1', 'shortcut', '')] = keras_format.format('conv4', 'block1',
                                                                                                     0, 'conv')
        # block3 unit_1-6 conv1-3
        # conv4 block1-6 1-3
        for i in range(1, 6):
            for j in range(1, 4):
                key = ckpt_format.format('block3', 'unit_%d' % i, 'conv%d' % j, '')
                value = keras_format.format('conv4', 'block%d' % i, j, 'conv')
                extractor_dict[key] = value
                key = ckpt_format.format('block3', 'unit_%d' % i, 'conv%d' % j, 'BatchNorm/')
                value = keras_format.format('conv4', 'block%d' % i, j, 'bn')
                extractor_dict[key] = value

        for tf_faster_rcnn_name_pre in extractor_dict.keys():
            if 'BatchNorm' in tf_faster_rcnn_name_pre:
                cur_weights = [
                    reader.get_tensor(tf_faster_rcnn_name_pre + 'gamma'),
                    reader.get_tensor(tf_faster_rcnn_name_pre + 'beta'),
                    reader.get_tensor(tf_faster_rcnn_name_pre + 'moving_mean'),
                    reader.get_tensor(tf_faster_rcnn_name_pre + 'moving_variance'),
                ]
            else:
                cur_weights = [
                    reader.get_tensor(tf_faster_rcnn_name_pre + 'weights'),
                    np.zeros(extractor.get_layer(name=extractor_dict[tf_faster_rcnn_name_pre]).variables[-1].shape)
                ]
            extractor.get_layer(name=extractor_dict[tf_faster_rcnn_name_pre]).set_weights(cur_weights)
            tf.logging.info('successfully loaded weights for {}'.format(extractor_dict[tf_faster_rcnn_name_pre]))

        rpn_head = self.get_layer('rpn_head')
        rpn_head_dict = {
            "build_rpn/rpn_conv/3x3/": "rpn_first_conv",
            "build_rpn/rpn_cls_score/": "rpn_score_conv",
            "build_rpn/rpn_bbox_pred/": "rpn_bbox_conv",
        }
        for tf_faster_rcnn_name_pre in rpn_head_dict.keys():
            rpn_head.get_layer(name=rpn_head_dict[tf_faster_rcnn_name_pre]).set_weights([
                reader.get_tensor(tf_faster_rcnn_name_pre + 'weights'),
                reader.get_tensor(tf_faster_rcnn_name_pre + 'biases'),
            ])
            tf.logging.info('successfully loaded weights for {}'.format(rpn_head_dict[tf_faster_rcnn_name_pre]))

        roi_head = self.get_layer('resnet_roi_head')
        roi_head_dict = {
            "Fast-RCNN/build_fc_layers/fc1/": "fc1",
            "Fast-RCNN/build_fc_layers/fc2/": "fc2",
            "Fast-RCNN/cls_fc/": "roi_head_score",
            "Fast-RCNN/reg_fc/": "roi_head_bboxes",
        }
        for tf_faster_rcnn_name_pre in roi_head_dict.keys():
            cur_weights = [
                reader.get_tensor(tf_faster_rcnn_name_pre + 'weights'),
                reader.get_tensor(tf_faster_rcnn_name_pre + 'biases'),
            ]
            roi_head.get_layer(name=roi_head_dict[tf_faster_rcnn_name_pre]).set_weights(cur_weights)
            tf.logging.info('successfully loaded weights for {}'.format(roi_head_dict[tf_faster_rcnn_name_pre]))

        fpn_neck = self.get_layer('resnet_fpn_neck')
        fpn_neck_dict = {
            "build_pyramid/build_P5/": "build_p5",
            "build_pyramid/build_P4/reduce_dim_P4/": "build_p4_reduce_dims",
            "build_pyramid/fuse_P4/": "build_p4",
            "build_pyramid/build_P3/reduce_dim_P3/": "build_p3_reduce_dims",
            "build_pyramid/fuse_P3/": "build_p3",
            "build_pyramid/build_P2/reduce_dim_P2/": "build_p2_reduce_dims",
            "build_pyramid/fuse_P2/": "build_p2",
        }
        for tf_faster_rcnn_name_pre in fpn_neck_dict.keys():
            cur_weights = [
                reader.get_tensor(tf_faster_rcnn_name_pre + 'weights'),
                reader.get_tensor(tf_faster_rcnn_name_pre + 'biases'),
            ]
            fpn_neck.get_layer(name=fpn_neck_dict[tf_faster_rcnn_name_pre]).set_weights(cur_weights)
            tf.logging.info('successfully loaded weights for {}'.format(fpn_neck_dict[tf_faster_rcnn_name_pre]))
