import tensorflow as tf

layers = tf.keras.layers
VGG_16_WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                              'releases/download/v0.1/'
                              'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

VGG_16_WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
RESNET_50_WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                                 'releases/download/v0.2/'
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')


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
                weights = reader.get_tensor(slim_tensor_name_pre+'weights')[:, :, ::-1, :]
                self.get_layer(name=slim_to_keras[slim_tensor_name_pre]).set_weights([
                    weights,
                    reader.get_tensor(slim_tensor_name_pre+'biases'),
                ])
            else:
                self.get_layer(name=slim_to_keras[slim_tensor_name_pre]).set_weights([
                    reader.get_tensor(slim_tensor_name_pre+'weights'),
                    reader.get_tensor(slim_tensor_name_pre+'biases'),
                ])
        tf.logging.info('successfully loaded slim vgg weights for vgg16 extractor.')
