import tensorflow as tf
from model.extractor.feature_extractor import Vgg16Extractor
from object_detection.dataset.pascal_tf_dataset_generator import get_dataset
from object_detection.model.faster_rcnn import RpnTrainingModel, \
    RoiTrainingModel, BaseRoiModel, BaseRpnModel
from object_detection.config.faster_rcnn_config import CONFIG
from tensorflow.contrib.summary import summary
from tensorflow.python.platform import tf_logging
from tqdm import tqdm

tf.enable_eager_execution()


def apply_gradients(model, optimizer, gradients):
    # for grad, var in zip(gradients, model.variables):
    #     if grad is not None:
    #         tf_logging.info(var.name)
    optimizer.apply_gradients(zip(gradients, model.variables),
                              global_step=tf.train.get_or_create_global_step())


def compute_gradients(model, loss, tape):
    return tape.gradient(loss, model.variables)


def train_step(model, loss, tape, optimizer):
    apply_gradients(model, optimizer, compute_gradients(model, loss, tape))


def _get_default_optimizer():
    lr = tf.train.exponential_decay(CONFIG['learning_rate_start'],
                                    tf.train.get_or_create_global_step(),
                                    CONFIG['learning_rate_decay_steps'],
                                    CONFIG['learning_rate_decay_rate'],
                                    True)
    return tf.train.MomentumOptimizer(lr, momentum=CONFIG['optimizer_momentum'])


def _get_training_dataset(preprocessing_type='caffe'):
    tf_records_list = ['/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_00.tfrecords',
                       '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_01.tfrecords',
                       '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_02.tfrecords',
                       '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_03.tfrecords',
                       '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_04.tfrecords', ]
    # tf_records_list = ['D:\\data\\VOCdevkit\\tf_eager_records\\pascal_train_00.tfrecords']
    return get_dataset(tf_records_list,
                       preprocessing_type=preprocessing_type,
                       min_size=CONFIG['image_min_size'], max_size=CONFIG['image_max_size'])


def _get_rpn_default_model(extractor):
    base_model = BaseRpnModel(
        ratios=CONFIG['ratios'],
        scales=CONFIG['scales'],
        extractor=extractor,
        extractor_stride=CONFIG['extractor_stride'],

        weight_decay=CONFIG['weight_decay'],
        rpn_proposal_num_pre_nms_train=CONFIG['rpn_proposal_train_pre_nms_sample_number'],
        rpn_proposal_num_post_nms_train=CONFIG['rpn_proposal_train_after_nms_sample_number'],
        rpn_proposal_num_pre_nms_test=CONFIG['rpn_proposal_test_pre_nms_sample_number'],
        rpn_proposal_num_post_nms_test=CONFIG['rpn_proposal_test_after_nms_sample_number'],
        rpn_proposal_nms_iou_threshold=CONFIG['rpn_proposal_nms_iou_threshold'],

        num_classes=CONFIG['num_classes'],
    )
    training_model = RpnTrainingModel(
        sigma=CONFIG['rpn_sigma'],
        rpn_training_pos_iou_threshold=CONFIG['rpn_pos_iou_threshold'],
        rpn_training_neg_iou_threshold=CONFIG['rpn_neg_iou_threshold'],
        rpn_training_total_num_samples=CONFIG['rpn_total_sample_number'],
        rpn_training_max_pos_samples=CONFIG['rpn_pos_sample_max_number'],
    )
    return base_model, training_model


def _get_roi_default_model():
    base_model = BaseRoiModel(
        extractor_stride=CONFIG['extractor_stride'],
        weight_decay=CONFIG['weight_decay'],
        roi_pool_size=CONFIG['roi_pooling_size'],
        num_classes=CONFIG['num_classes'],
        roi_head_keep_dropout_rate=CONFIG['roi_head_keep_dropout_rate'],
    )
    training_model = RoiTrainingModel(
        num_classes=CONFIG['num_classes'],
        sigma=CONFIG['roi_sigma'],
        roi_training_pos_iou_threshold=CONFIG['roi_pos_iou_threshold'],
        roi_training_neg_iou_threshold=CONFIG['roi_neg_iou_threshold'],
        roi_training_total_num_samples=CONFIG['roi_total_sample_number'],
        roi_training_max_pos_samples=CONFIG['roi_pos_sample_max_number']
    )
    return base_model, training_model


def train_one_epoch(rpn_model, rpn_training_model,
                    roi_model, roi_training_model,
                    dataset, optimizer):
    for idx, (image, gt_bboxes, gt_labels, _) in tqdm(enumerate(dataset)):
        gt_bboxes = tf.squeeze(gt_bboxes, axis=0)
        gt_labels = tf.to_int32(tf.squeeze(gt_labels, axis=0))
        tf_logging.info()

        with tf.GradientTape() as tape:
            image_shape, anchors, rpn_score, rpn_bboxes_txtytwth, rpn_proposals_bboxes, shared_features = rpn_model(
                image, True)
            rpn_cls_loss, rpn_reg_loss = rpn_training_model((image_shape, anchors, rpn_score,
                                                             rpn_bboxes_txtytwth, gt_bboxes), True)
            tf.contrib.summary.scalar("rpn_cls_loss", rpn_cls_loss)
            tf.contrib.summary.scalar("rpn_reg_loss", rpn_reg_loss)
            tf_logging.info('rpn loss', rpn_cls_loss.numpy(), rpn_reg_loss.numpy())

            train_step(rpn_model, rpn_cls_loss + rpn_reg_loss, tape, optimizer)

        with tf.GradientTape() as tape:
            roi_score, roi_bboxes_txtytwth = roi_model((shared_features, rpn_proposals_bboxes), True)
            roi_cls_loss, roi_reg_loss = roi_training_model((image_shape, rpn_proposals_bboxes,
                                                             roi_score, roi_bboxes_txtytwth,
                                                             gt_bboxes, gt_labels), True)
            tf.contrib.summary.scalar("roi_cls_loss", roi_cls_loss)
            tf.contrib.summary.scalar("roi_reg_loss", roi_reg_loss)
            tf_logging.info('roi loss', roi_cls_loss.numpy(), roi_reg_loss.numpy())

            train_step(roi_model, roi_cls_loss + roi_reg_loss, tape, optimizer)


def train(rpn_model, rpn_training_model, roi_model, roi_training_model, dataset, optimizer,
          train_dir='/home/tensorflow05/zyy/tf_eager_object_detection/logs-alt',
          val_dir='/home/tensorflow05/zyy/tf_eager_object_detection/logs-alt/val',
          # train_dir='E:\\PycharmProjects\\tf_eager_object_detection\\logs',
          # val_dir='E:\\PycharmProjects\\tf_eager_object_detection\\logs\\val',
          ):
    tf.train.get_or_create_global_step()
    train_writer = tf.contrib.summary.create_file_writer(train_dir, flush_millis=100000)

    for i in range(CONFIG['epochs']):
        tf_logging.info('epoch %d starting...' % (i + 1))
        with train_writer.as_default(), summary.record_summaries_every_n_global_steps(50):
            train_one_epoch(rpn_model, rpn_training_model, roi_model, roi_training_model,
                            dataset, optimizer)


if __name__ == '__main__':
    vgg16_extractor = Vgg16Extractor()
    base_rpn_model, rpn_training = _get_rpn_default_model(vgg16_extractor)
    base_roi_model, roi_training = _get_roi_default_model()
    train(base_rpn_model, rpn_training,
          base_roi_model, roi_training,
          _get_training_dataset(), _get_default_optimizer())
