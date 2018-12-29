import tensorflow as tf


class BaseExtractor(tf.keras.Model):
    def __init__(self, base_extractor):
        super().__init__()
        self._extractor = base_extractor

    def call(self, inputs, training=None, mask=None):
        return self._extractor(inputs, training, mask)


class Vgg16Extractor(BaseExtractor):
    def __init__(self):
        extractor = tf.keras.applications.VGG16(include_top=False,
                                                weights='imagenet',
                                                input_tensor=None,
                                                input_shape=None,
                                                pooling=None,
                                                classes=None, )
        super().__init__(base_extractor=extractor)


class ResNet50Extractor(BaseExtractor):
    def __init__(self):
        extractor = tf.keras.applications.ResNet50(include_top=False,
                                                   weights='imagenet',
                                                   input_tensor=None,
                                                   input_shape=None,
                                                   pooling=None,
                                                   classes=None, )
        super().__init__(base_extractor=extractor)
