import tensorflow as tf
from object_detection.model.faster_rcnn.base_faster_rcnn_model import BaseFasterRcnn

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
           conv_shortcut=True, name=None):
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
                                 name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME',
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
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
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def get_resnet_model(stack_fn,
                     preact,
                     use_bias,
                     model_name='resnet',
                     ):
    img_input = layers.Input(shape=(None, None, 3))
    bn_axis = 3

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if preact is False:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact is True:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    # Create model.
    model = tf.keras.Model(img_input, x, name=model_name)

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


def get_resnet_v1_extractor(depth):
    if depth == 50:
        def stack_fn(x):
            x = stack1(x, 64, 3, stride1=1, name='conv2')
            x = stack1(x, 128, 4, name='conv3')
            x = stack1(x, 256, 6, name='conv4')
            return x
    elif depth == 101:
        def stack_fn(x):
            x = stack1(x, 64, 3, stride1=1, name='conv2')
            x = stack1(x, 128, 4, name='conv3')
            x = stack1(x, 256, 23, name='conv4')
            return x
    elif depth == 152:
        def stack_fn(x):
            x = stack1(x, 64, 3, stride1=1, name='conv2')
            x = stack1(x, 128, 8, name='conv3')
            x = stack1(x, 256, 36, name='conv4')
            return x
    else:
        raise ValueError('unknown depth {}'.format(depth))

    return get_resnet_model(stack_fn, False, True, 'resnet{}'.format(depth))


def get_resnet_v1_roi_head(depth, roi_feature_size, num_classes, weight_decay=.0):
    if depth not in [50, 101, 152]:
        raise ValueError('unknown depth {}'.format(depth))
    model_name = 'resnet{}'.format(depth)

    features_input = layers.Input(roi_feature_size)
    x = stack1(features_input, 512, 3, name='conv5')
    x = layers.GlobalAveragePooling2D()(x)
    score = layers.Dense(num_classes, name='roi_head_score', activation=None,
                         kernel_initializer=tf.random_normal_initializer(0, 0.01),
                         kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    bboxes = layers.Dense(4 * num_classes, name='roi_head_bboxes', activation=None,
                          kernel_initializer=tf.random_normal_initializer(0, 0.001),
                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    roi_head_model = tf.keras.Model(features_input, [score, bboxes], name='{}_roi_head'.format(model_name))
    if model_name in WEIGHTS_HASHES:
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
        file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = tf.keras.utils.get_file(file_name,
                                               BASE_WEIGHTS_PATH + file_name,
                                               cache_subdir='models',
                                               file_hash=file_hash)
        roi_head_model.load_weights(weights_path, by_name=True)
        tf.logging.info('successfully load keras pre-trained weights for {} roi head'.format(model_name))
    return roi_head_model


class ResNetFasterRcnn(BaseFasterRcnn):
    def __init__(self,
                 # resnet 特有参数
                 depth=50,
                 roi_feature_size=(7, 7, 1024),

                 # 通用参数
                 num_classes=21,
                 weight_decay=0.0001,
                 ratios=(0.5, 1.0, 2.0),
                 scales=(8, 16, 32),
                 extractor_stride=16,

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
                 prediction_score_threshold=0.3,
                 ):
        if depth not in [50, 101, 152]:
            raise ValueError('unknown resnet layers number {}'.format(depth))

        self._depth = depth
        self._roi_feature_size = roi_feature_size

        super().__init__(num_classes=num_classes,
                         weight_decay=weight_decay,

                         ratios=ratios,
                         scales=scales,
                         extractor_stride=extractor_stride,

                         rpn_proposal_means=rpn_proposal_means,
                         rpn_proposal_stds=rpn_proposal_stds,

                         rpn_proposal_num_pre_nms_train=rpn_proposal_num_pre_nms_train,
                         rpn_proposal_num_post_nms_train=rpn_proposal_num_post_nms_train,
                         rpn_proposal_num_pre_nms_test=rpn_proposal_num_pre_nms_test,
                         rpn_proposal_num_post_nms_test=rpn_proposal_num_post_nms_test,
                         rpn_proposal_nms_iou_threshold=rpn_proposal_nms_iou_threshold,

                         rpn_sigma=rpn_sigma,
                         rpn_training_pos_iou_threshold=rpn_training_pos_iou_threshold,
                         rpn_training_neg_iou_threshold=rpn_training_neg_iou_threshold,
                         rpn_training_total_num_samples=rpn_training_total_num_samples,
                         rpn_training_max_pos_samples=rpn_training_max_pos_samples,

                         roi_proposal_means=roi_proposal_means,
                         roi_proposal_stds=roi_proposal_stds,

                         roi_pool_size=roi_pool_size,

                         roi_sigma=roi_sigma,
                         roi_training_pos_iou_threshold=roi_training_pos_iou_threshold,
                         roi_training_neg_iou_threshold=roi_training_neg_iou_threshold,
                         roi_training_total_num_samples=roi_training_total_num_samples,
                         roi_training_max_pos_samples=roi_training_max_pos_samples,

                         prediction_max_objects_per_image=prediction_max_objects_per_image,
                         prediction_max_objects_per_class=prediction_max_objects_per_class,
                         prediction_nms_iou_threshold=prediction_nms_iou_threshold,
                         prediction_score_threshold=prediction_score_threshold,
                         )

    def _get_roi_head(self):
        return get_resnet_v1_roi_head(depth=self._depth,
                                      roi_feature_size=self._roi_feature_size,
                                      num_classes=self.num_classes,
                                      weight_decay=self.weight_decay)

    def _get_extractor(self):
        return get_resnet_v1_extractor(depth=self._depth)
