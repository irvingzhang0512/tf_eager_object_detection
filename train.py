import tensorflow as tf
from object_detection.model.feature_extractor import Vgg16Extractor
from object_detection.dataset.pascal_tf_dataset_generator import get_dataset
from object_detection.model.faster_rcnn import BaseFasterRcnnModel, FasterRcnnTrainingModel, RpnTrainingModel, \
    RoiTrainingModel
from object_detection.config.faster_rcnn_config import CONFIG

tf.enable_eager_execution()


def apply_gradients(model, optimizer, gradients):
    optimizer.apply_gradients(zip(gradients, model.variables))


def compute_gradients(model, loss, tape):
    return tape.gradient(loss, model.variables)


def train_step(model, loss, tape, optimizer):
    apply_gradients(model, optimizer, compute_gradients(model, loss, tape))


def train(dataset, faster_rcnn_model, optimizer):
    base_model, training_model = faster_rcnn_model
    for image, gt_bboxes, gt_labels, gt_labels_text in dataset:
        with tf.GradientTape() as tape:
            image_shape, anchors, rpn_score, rpn_bboxes_txtytwth, rpn_proposals_bboxes, roi_score, roi_bboxes_txtytwth = base_model(
                image)
            rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss = training_model((gt_bboxes, gt_labels,
                                                                                     image_shape, anchors,
                                                                                     rpn_score, rpn_bboxes_txtytwth,
                                                                                     rpn_proposals_bboxes,
                                                                                     roi_score, roi_bboxes_txtytwth),
                                                                                    True)
            train_step(faster_rcnn_model,
                       rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss,
                       tape,
                       optimizer)
            print(rpn_cls_loss.numpy(), rpn_reg_loss.numpy(), roi_cls_loss.numpy(), roi_reg_loss.numpy())
            print(rpn_cls_loss.numpy() + rpn_reg_loss.numpy() + roi_cls_loss.numpy() + roi_reg_loss.numpy())


def _get_default_base_model():
    base_model = BaseFasterRcnnModel(
        ratios=CONFIG['ratios'],
        scales=CONFIG['scales'],
        extractor=Vgg16Extractor(),
        extractor_stride=CONFIG['extractor_stride'],

        rpn_proposal_num_pre_nms_train=12000,
        rpn_proposal_num_post_nms_train=2000,
        rpn_proposal_num_pre_nms_test=6000,
        rpn_proposal_num_post_nms_test=300,
        rpn_proposal_nms_iou_threshold=0.7,

        roi_pool_size=7,
        num_classes=21,
        roi_head_keep_dropout_rate=0.5,
    )
    training_model = FasterRcnnTrainingModel(
        rpn_training_model=RpnTrainingModel(
            cls_loss_weight=3,
            reg_loss_weight=1,
            rpn_training_pos_iou_threshold=0.7,
            rpn_training_neg_iou_threshold=0.3,
            rpn_training_total_num_samples=256,
            rpn_training_max_pos_samples=128,
        ),
        roi_training_model=RoiTrainingModel(
            num_classes=21,
            cls_loss_weight=3,
            reg_loss_weight=1,
            roi_training_pos_iou_threshold=0.5,
            roi_training_neg_iou_threshold=0.1,
            roi_training_total_num_samples=128,
            roi_training_max_pos_samples=32
        ),
    )
    return base_model, training_model


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
