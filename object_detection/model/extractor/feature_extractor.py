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
    def __init__(self, ckpt_file_path=None):
        super().__init__(name='vgg16')
        self._ckpt_file_path = ckpt_file_path
        # Block 1
        self.add(layers.Conv2D(64, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block1_conv1', trainable=False,
                               input_shape=(None, None, 3)))
        self.add(layers.Conv2D(64, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block1_conv2', trainable=False))
        self.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', padding='same'))

        # Block 2
        self.add(layers.Conv2D(128, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block2_conv1', trainable=False))
        self.add(layers.Conv2D(128, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block2_conv2', trainable=False))
        self.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', padding='same'))

        # Block 3
        self.add(layers.Conv2D(256, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block3_conv1'))
        self.add(layers.Conv2D(256, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block3_conv2'))
        self.add(layers.Conv2D(256, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block3_conv3'))
        self.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same'))

        # Block 4
        self.add(layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block4_conv1'))
        self.add(layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block4_conv2'))
        self.add(layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block4_conv3'))
        self.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', padding='same'))

        # Block 5
        self.add(layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block5_conv1'))
        self.add(layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block5_conv2'))
        self.add(layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block5_conv3'))
        if ckpt_file_path:
            self.load_slim_weights(ckpt_file_path)
        else:
            self._load_keras_weights()

    def _load_keras_weights(self):
        weights_path = tf.keras.utils.get_file(
            'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
            VGG_16_WEIGHTS_PATH,
            cache_subdir='models',
            file_hash='64373286793e3c8b2b4e3219cbf3544b')
        self.load_weights(weights_path, by_name=True)
        tf.logging.info('successfully loaded keras vgg weights.')

    def load_slim_weights(self, ckpt_file_path):
        reader = tf.train.NewCheckpointReader(ckpt_file_path)
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
            self.get_layer(name=slim_to_keras[slim_tensor_name_pre]).set_weights([
                reader.get_tensor(slim_tensor_name_pre+'weights'),
                reader.get_tensor(slim_tensor_name_pre+'biases'),
            ])
        tf.logging.info('successfully loaded slim vgg weights.')


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


class ResNet50Extractor(tf.keras.Model):
    def __init__(self):
        super().__init__()

        img_input = layers.Input(shape=(None, None, 3))
        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
        x = layers.Conv2D(64, (7, 7),
                          strides=(2, 2),
                          padding='valid',
                          kernel_initializer='he_normal',
                          name='conv1')(x)
        x = layers.BatchNormalization(axis=3, name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        # x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        model = tf.keras.Model(img_input, x, name='resnet50')

        weights_path = tf.keras.utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            RESNET_50_WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        self._model = model

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs, training, mask)
