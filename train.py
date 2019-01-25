import tensorflow as tf
from object_detection.model.feature_extractor import Vgg16Extractor
from object_detection.model.faster_rcnn import BaseFasterRcnn, FasterRcnnEnd2EndTrainingModel
from object_detection.dataset.pascal_tf_dataset_generator import get_dataset
from object_detection.model.rpn import RpnTrainingModel
from object_detection.model.roi import RoiTrainingModel

tf.enable_eager_execution()


def apply_gradients(model, optimizer, gradients):
    optimizer.apply_gradients(zip(gradients, model.variables))


def compute_gradients(model, loss, tape):
    return tape.gradient(loss, model.variables)


def train_step(model, loss, tape, optimizer):
    apply_gradients(model, optimizer, compute_gradients(model, loss, tape))


def train(dataset, faster_rcnn_model, optimizer):
    for images, gt_bboxes, gt_labels, gt_labels_text in dataset:
        gt_bboxes = tf.squeeze(gt_bboxes, axis=0)
        gt_labels = tf.squeeze(gt_labels, axis=0)
        gt_labels_text = tf.squeeze(gt_labels_text, axis=0)
        with tf.GradientTape() as tape:
            rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss = faster_rcnn_model((images, gt_bboxes, gt_labels),
                                                                                       True)
            train_step(faster_rcnn_model,
                       rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss,
                       tape,
                       optimizer)
            print(rpn_cls_loss.numpy(), rpn_reg_loss.numpy(), roi_cls_loss.numpy(), roi_reg_loss.numpy())
            print(rpn_cls_loss.numpy() + rpn_reg_loss.numpy() + roi_cls_loss.numpy() + roi_reg_loss.numpy())


def _get_default_base_model():
    return FasterRcnnEnd2EndTrainingModel(
        base_model=BaseFasterRcnn(
            ratios=[0.5, 1, 2],
            scales=[8 * 16, 16 * 16, 32 * 16],
            extractor=Vgg16Extractor(),
            extractor_stride=16,
        ),
        rpn_training_model=RpnTrainingModel(),
        roi_training_model=RoiTrainingModel(),
    )


def _get_default_optimizer():
    return tf.train.MomentumOptimizer(0.001, 0.9)


def _get_default_dataset():
    tf_records_list = ['/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_00.tfrecords',
                       '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_01.tfrecords',
                       '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_02.tfrecords',
                       '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_03.tfrecords',
                       '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_04.tfrecords', ]
    return get_dataset(tf_records_list, repeat=100)


if __name__ == '__main__':
    cur_base_model = _get_default_base_model()
    cur_dataset = _get_default_dataset()
    cur_optimizer = _get_default_optimizer()
    train(cur_dataset, cur_base_model, cur_optimizer)
