import tensorflow as tf
from object_detection.model.faster_rcnn.base_faster_rcnn_model import BaseFasterRcnn

__all__ = ['Vgg16FasterRcnn']
layers = tf.keras.layers
VGG_16_WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels.h5')


class Vgg16FasterRcnn(BaseFasterRcnn):
    def __init__(self,
                 # Vgg16FasterRcnn 特有参数
                 slim_ckpt_file_path=None,
                 roi_head_keep_dropout_rate=0.5,
                 roi_feature_size=(7, 7, 512),

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
                 prediction_score_threshold=0.3, ):
        self._slim_ckpt_file_path = slim_ckpt_file_path
        self._roi_feature_size = roi_feature_size
        self._roi_head_keep_dropout_rate = roi_head_keep_dropout_rate
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
                         roi_pooling_max_pooling_flag=roi_pooling_max_pooling_flag,

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
        return Vgg16RoiHead(self.num_classes,
                            roi_feature_size=self._roi_feature_size,
                            keep_rate=self._roi_head_keep_dropout_rate,
                            weight_decay=self.weight_decay,
                            slim_ckpt_file_path=self._slim_ckpt_file_path)

    def _get_extractor(self):
        return Vgg16Extractor(weight_decay=self.weight_decay,
                              slim_ckpt_file_path=self._slim_ckpt_file_path)

    def load_tf_faster_rcnn_tf_weights(self, ckpt_file_path):
        reader = tf.train.load_checkpoint(ckpt_file_path)
        extractor = self.get_layer('vgg16')
        extractor_dict = {
            "vgg_16/conv1/conv1_1/": "block1_conv1",
            "vgg_16/conv1/conv1_2/": "block1_conv2",

            "vgg_16/conv2/conv2_1/": "block2_conv1",
            "vgg_16/conv2/conv2_2/": "block2_conv2",

            "vgg_16/conv3/conv3_1/": "block3_conv1",
            "vgg_16/conv3/conv3_2/": "block3_conv2",
            "vgg_16/conv3/conv3_3/": "block3_conv3",

            "vgg_16/conv4/conv4_1/": "block4_conv1",
            "vgg_16/conv4/conv4_2/": "block4_conv2",
            "vgg_16/conv4/conv4_3/": "block4_conv3",

            "vgg_16/conv5/conv5_1/": "block5_conv1",
            "vgg_16/conv5/conv5_2/": "block5_conv2",
            "vgg_16/conv5/conv5_3/": "block5_conv3",
        }
        for slim_tensor_name_pre in extractor_dict.keys():
            extractor.get_layer(name=extractor_dict[slim_tensor_name_pre]).set_weights([
                reader.get_tensor(slim_tensor_name_pre + 'weights'),
                reader.get_tensor(slim_tensor_name_pre + 'biases'),
            ])
            tf.logging.info('successfully loaded weights for {}'.format(extractor_dict[slim_tensor_name_pre]))

        rpn_head = self.get_layer('vgg16_rpn_head')
        rpn_head_dict = {
            'vgg_16/rpn_conv/3x3/': 'rpn_first_conv',
            'vgg_16/rpn_cls_score/': 'rpn_score_conv',
            'vgg_16/rpn_bbox_pred/': 'rpn_bbox_conv',
        }
        for slim_tensor_name_pre in rpn_head_dict.keys():
            rpn_head.get_layer(rpn_head_dict[slim_tensor_name_pre]).set_weights([
                reader.get_tensor(slim_tensor_name_pre + 'weights'),
                reader.get_tensor(slim_tensor_name_pre + 'biases')
            ])
            tf.logging.info('successfully loaded weights for {}'.format(rpn_head_dict[slim_tensor_name_pre]))

        roi_head = self.get_layer('vgg16_roi_head')
        roi_head_dict = {
            'vgg_16/fc6/': 'fc1',
            'vgg_16/fc7/': 'fc2',
            'vgg_16/bbox_pred/': 'roi_head_bboxes',
            'vgg_16/cls_score/': 'roi_head_score'
        }
        for slim_tensor_name_pre in roi_head_dict.keys():
            roi_head.get_layer(roi_head_dict[slim_tensor_name_pre]).set_weights([
                reader.get_tensor(slim_tensor_name_pre + 'weights'),
                reader.get_tensor(slim_tensor_name_pre + 'biases')
            ])
            tf.logging.info('successfully loaded weights for {}'.format(roi_head_dict[slim_tensor_name_pre]))

    def disable_biases(self):
        # vgg16 doesn't need to diable biases
        pass


class Vgg16RoiHead(tf.keras.Model):
    def __init__(self, num_classes,
                 roi_feature_size=(7, 7, 512),
                 keep_rate=0.5, weight_decay=0.0005,
                 slim_ckpt_file_path=None, ):
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
        self._flatten_layer = layers.Flatten()

        self.build((None, *roi_feature_size))

        if slim_ckpt_file_path is None:
            self._load_keras_weights()
        else:
            self._load_slim_weights(slim_ckpt_file_path)

    def _load_slim_weights(self, ckpt_file_path):
        reader = tf.train.NewCheckpointReader(ckpt_file_path)
        slim_to_keras = {
            "vgg_16/fc6/": "fc1",
            "vgg_16/fc7/": "fc2",
        }

        for slim_tensor_name_pre in slim_to_keras.keys():
            cur_layer = self.get_layer(name=slim_to_keras[slim_tensor_name_pre])
            cur_layer.set_weights([
                reader.get_tensor(slim_tensor_name_pre + 'weights').reshape(
                    cur_layer.variables[0].get_shape().as_list()),
                reader.get_tensor(slim_tensor_name_pre + 'biases').reshape(
                    cur_layer.variables[1].get_shape().as_list()),
            ])
        tf.logging.info('successfully loaded slim vgg weights for roi head.')

    def _load_keras_weights(self):
        weights_path = tf.keras.utils.get_file(
            'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
            VGG_16_WEIGHTS_PATH,
            cache_subdir='models',
            file_hash='64373286793e3c8b2b4e3219cbf3544b')
        self.load_weights(weights_path, by_name=True)
        tf.logging.info('successfully load pretrained weights for roi head.')

    def call(self, inputs, training=None):
        """
        输入 roi pooling 的结果
        对每个 roi pooling 的结果进行预测（预测bboxes）
        :param inputs:  roi_features, [num_rois, pool_size, pool_size, num_channels]
        :param training:
        :param mask:
        :return:
        """
        x = self._flatten_layer(inputs)
        x = self._fc1(x)
        x = self._dropout1(x, training)
        x = self._fc2(x)
        x = self._dropout2(x, training)
        score = self._score_layer(x)
        bboxes = self._roi_bboxes_layer(x)

        return score, bboxes


class Vgg16Extractor(tf.keras.Sequential):
    def __init__(self, weight_decay=0.0001,
                 slim_ckpt_file_path=None):
        super().__init__(name='vgg16')
        # Block 1
        self.add(layers.Conv2D(64, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block1_conv1', trainable=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               input_shape=(None, None, 3)))
        self.add(layers.Conv2D(64, (3, 3),
                               activation='relu',
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               name='block1_conv2', trainable=False))
        self.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', padding='same'))

        # Block 2
        self.add(layers.Conv2D(128, (3, 3),
                               activation='relu',
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               name='block2_conv1', trainable=False))
        self.add(layers.Conv2D(128, (3, 3),
                               activation='relu',
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               name='block2_conv2', trainable=False))
        self.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', padding='same'))

        # Block 3
        self.add(layers.Conv2D(256, (3, 3),
                               activation='relu',
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               name='block3_conv1'))
        self.add(layers.Conv2D(256, (3, 3),
                               activation='relu',
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               name='block3_conv2'))
        self.add(layers.Conv2D(256, (3, 3),
                               activation='relu',
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               name='block3_conv3'))
        self.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same'))

        # Block 4
        self.add(layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               name='block4_conv1'))
        self.add(layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               name='block4_conv2'))
        self.add(layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               name='block4_conv3'))
        self.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', padding='same'))

        # Block 5
        self.add(layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               name='block5_conv1'))
        self.add(layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               name='block5_conv2'))
        self.add(layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                               name='block5_conv3'))
        if slim_ckpt_file_path:
            self.load_slim_weights(slim_ckpt_file_path)
        else:
            self._load_keras_weights()

    def _load_keras_weights(self):
        weights_path = tf.keras.utils.get_file(
            'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
            VGG_16_WEIGHTS_PATH,
            cache_subdir='models',
            file_hash='64373286793e3c8b2b4e3219cbf3544b')
        self.load_weights(weights_path, by_name=True)
        tf.logging.info('successfully loaded keras vgg weights for vgg16 extractor.')

    def load_slim_weights(self, slim_ckpt_file_path):
        reader = tf.train.NewCheckpointReader(slim_ckpt_file_path)
        slim_to_keras = {
            "vgg_16/conv1/conv1_1/": "block1_conv1",
            "vgg_16/conv1/conv1_2/": "block1_conv2",

            "vgg_16/conv2/conv2_1/": "block2_conv1",
            "vgg_16/conv2/conv2_2/": "block2_conv2",

            "vgg_16/conv3/conv3_1/": "block3_conv1",
            "vgg_16/conv3/conv3_2/": "block3_conv2",
            "vgg_16/conv3/conv3_3/": "block3_conv3",

            "vgg_16/conv4/conv4_1/": "block4_conv1",
            "vgg_16/conv4/conv4_2/": "block4_conv2",
            "vgg_16/conv4/conv4_3/": "block4_conv3",

            "vgg_16/conv5/conv5_1/": "block5_conv1",
            "vgg_16/conv5/conv5_2/": "block5_conv2",
            "vgg_16/conv5/conv5_3/": "block5_conv3",
        }
        for slim_tensor_name_pre in slim_to_keras.keys():
            if slim_tensor_name_pre == 'vgg_16/conv1/conv1_1/':
                weights = reader.get_tensor(slim_tensor_name_pre + 'weights')[:, :, ::-1, :]
                self.get_layer(name=slim_to_keras[slim_tensor_name_pre]).set_weights([
                    weights,
                    reader.get_tensor(slim_tensor_name_pre + 'biases'),
                ])
            else:
                self.get_layer(name=slim_to_keras[slim_tensor_name_pre]).set_weights([
                    reader.get_tensor(slim_tensor_name_pre + 'weights'),
                    reader.get_tensor(slim_tensor_name_pre + 'biases'),
                ])
        tf.logging.info('successfully loaded slim vgg weights for vgg16 extractor.')
