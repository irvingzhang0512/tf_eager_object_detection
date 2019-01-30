import tensorflow as tf
from object_detection.model.feature_extractor import Vgg16Extractor
from object_detection.dataset.pascal_tf_dataset_generator import get_dataset
from object_detection.model.faster_rcnn import BaseFasterRcnnModel, FasterRcnnTrainingModel, RpnTrainingModel, \
    RoiTrainingModel, post_ops_prediction
from object_detection.config.faster_rcnn_config import CONFIG
from object_detection.utils.pascal_voc_map_utils import eval_detection_voc
import tensorflow.contrib.eager as tfe
import time
from tensorflow.contrib.summary import summary as eager_summary

tf.enable_eager_execution()


def apply_gradients(model, optimizer, gradients):
    optimizer.apply_gradients(zip(gradients, model.variables),
                              global_step=tf.train.get_or_create_global_step())


def compute_gradients(model, loss, tape):
    return tape.gradient(loss, model.variables)


def train_step(model, loss, tape, optimizer):
    apply_gradients(model, optimizer, compute_gradients(model, loss, tape))


def _get_default_base_model():
    base_model = BaseFasterRcnnModel(
        ratios=CONFIG['ratios'],
        scales=CONFIG['scales'],
        extractor=Vgg16Extractor(),
        extractor_stride=CONFIG['extractor_stride'],

        weight_decay=CONFIG['weight_decay'],
        rpn_proposal_num_pre_nms_train=CONFIG['rpn_proposal_train_pre_nms_sample_number'],
        rpn_proposal_num_post_nms_train=CONFIG['rpn_proposal_train_after_nms_sample_number'],
        rpn_proposal_num_pre_nms_test=CONFIG['rpn_proposal_test_pre_nms_sample_number'],
        rpn_proposal_num_post_nms_test=CONFIG['rpn_proposal_test_after_nms_sample_number'],
        rpn_proposal_nms_iou_threshold=CONFIG['rpn_proposal_nms_iou_threshold'],

        roi_pool_size=CONFIG['roi_pooling_size'],
        num_classes=CONFIG['num_classes'],
        roi_head_keep_dropout_rate=CONFIG['roi_head_keep_dropout_rate'],
    )
    training_model = FasterRcnnTrainingModel(
        rpn_training_model=RpnTrainingModel(
            cls_loss_weight=CONFIG['rpn_cls_loss_weight'],
            reg_loss_weight=CONFIG['rpn_reg_loss_weight'],
            rpn_training_pos_iou_threshold=CONFIG['rpn_pos_iou_threshold'],
            rpn_training_neg_iou_threshold=CONFIG['rpn_neg_iou_threshold'],
            rpn_training_total_num_samples=CONFIG['rpn_total_sample_number'],
            rpn_training_max_pos_samples=CONFIG['rpn_pos_sample_max_number'],
        ),
        roi_training_model=RoiTrainingModel(
            num_classes=CONFIG['num_classes'],
            cls_loss_weight=CONFIG['roi_cls_loss_weight'],
            reg_loss_weight=CONFIG['roi_reg_loss_weight'],
            roi_training_pos_iou_threshold=CONFIG['roi_pos_iou_threshold'],
            roi_training_neg_iou_threshold=CONFIG['roi_neg_iou_threshold'],
            roi_training_total_num_samples=CONFIG['roi_total_sample_number'],
            roi_training_max_pos_samples=CONFIG['roi_pos_sample_max_number']
        ),
    )
    return base_model, training_model


def _get_default_optimizer():
    return tf.train.MomentumOptimizer(CONFIG['learning_rate_start'], momentum=CONFIG['optimizer_momentum'])


def _get_training_dataset():
    tf_records_list = ['/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_00.tfrecords',
                       '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_01.tfrecords',
                       '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_02.tfrecords',
                       '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_03.tfrecords',
                       '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_train_04.tfrecords', ]
    return get_dataset(tf_records_list,
                       min_size=CONFIG['image_min_size'], max_size=CONFIG['image_max_size'])


def _get_evaluating_dataset():
    tf_records_list = ['/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_val_00.tfrecords',
                       '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_val_01.tfrecords',
                       '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_val_02.tfrecords',
                       '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_val_03.tfrecords',
                       '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_val_04.tfrecords', ]
    return get_dataset(tf_records_list,
                       min_size=CONFIG['image_min_size'], max_size=CONFIG['image_max_size'],
                       argument=False, )


def train_one_epoch(dataset, base_model, training_model, optimizer, logs_every_n_steps=1):
    for idx, (image, gt_bboxes, gt_labels, _) in enumerate(dataset):
        with tf.GradientTape() as tape:
            shape, anchors, rpn_score, rpn_txtytwth, rpn_proposals, roi_score, roi_txtytwth = base_model(image,
                                                                                                         True)
            rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss = training_model((gt_bboxes, gt_labels,
                                                                                     shape, anchors,
                                                                                     rpn_score, rpn_txtytwth,
                                                                                     rpn_proposals,
                                                                                     roi_score, roi_txtytwth),
                                                                                    True)
            l2_loss = tf.add_n(base_model.losses)
            total_loss = rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss + l2_loss

            tf.contrib.summary.graph(tf.get_default_graph())
            tf.contrib.summary.scalar("rpn_cls_loss", rpn_cls_loss)
            tf.contrib.summary.scalar("rpn_reg_loss", rpn_reg_loss)
            tf.contrib.summary.scalar("roi_cls_loss", roi_cls_loss)
            tf.contrib.summary.scalar("roi_reg_loss", roi_reg_loss)
            tf.contrib.summary.scalar("l2_loss", l2_loss)
            tf.contrib.summary.scalar("total_loss", total_loss)

            train_step(base_model, total_loss, tape, optimizer)
        if idx % logs_every_n_steps == 0:
            print('steps %d loss: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (idx + 1,
                                                                         rpn_cls_loss.numpy(), rpn_reg_loss.numpy(),
                                                                         roi_cls_loss.numpy(), roi_reg_loss.numpy(),
                                                                         tf.add_n(base_model.losses), total_loss)
                  )


def train(training_dataset, evaluating_dataset, base_model, training_model, optimizer,
          train_dir='./logs', val_dir='./logs/val'):
    tf.train.get_or_create_global_step()
    # ckpt = tf.train.Checkpoint(model=base_model)
    # ckpt.save(file_prefix='./logs/ckpt')
    # saver = tfe.Saver(base_model.variables)
    # saver = tf.train.Saver(base_model.variables)
    # saver.save('./logs/model.ckpt')
    train_writer = tf.contrib.summary.create_file_writer(train_dir, flush_millis=10000)
    val_writer = tf.contrib.summary.create_file_writer(val_dir, flush_millis=10000)

    for i in range(CONFIG['epochs']):
        print('epoch %d starting...' % (i + 1))
        start = time.time()
        with train_writer.as_default(), tf.contrib.summary.always_record_summaries():
            train_one_epoch(training_dataset, base_model, training_model, optimizer)
        train_end = time.time()
        print('epoch %d training finished, costing %d seconds, start evaluating...' % (i + 1, train_end - start))
        with val_writer.as_default(), tf.contrib.summary.always_record_summaries():
            res = evaluate(evaluating_dataset, cur_base_model)
        print('epoch %d evaluating finished, costing %d seconds, current mAP is %.4f' % (i + 1,
                                                                                         time.time() - train_end,
                                                                                         res['map']))


def evaluate(dataset, base_faster_rcnn_model, use_07_metric=False):
    gt_bboxes = []
    gt_labels = []
    pred_bboxes = []
    pred_labels = []
    pred_scores = []

    for cur_image, cur_gt_bboxes, cur_gt_labels, _ in dataset:
        cur_gt_bboxes = tf.squeeze(cur_gt_bboxes, axis=0)
        cur_gt_labels = tf.to_int32(tf.squeeze(cur_gt_labels, axis=0))

        image_shape, _, _, _, rpn_proposals_bboxes, roi_score, roi_bboxes_txtytwth = base_faster_rcnn_model(cur_image,
                                                                                                            False)
        cur_pred_bboxes, cur_pred_labels, cur_pred_scores = post_ops_prediction((rpn_proposals_bboxes,
                                                                                 roi_score, roi_bboxes_txtytwth,
                                                                                 image_shape),
                                                                                num_classes=CONFIG['num_classes'],
                                                                                max_num_per_class=CONFIG[
                                                                                    'max_objects_per_class_per_image'],
                                                                                max_num_per_image=CONFIG[
                                                                                    'max_objects_per_image'],
                                                                                nms_iou_threshold=CONFIG[
                                                                                    'predictions_nms_iou_threshold'],
                                                                                )
        gt_bboxes.append(cur_gt_bboxes.numpy())
        gt_labels.append(cur_gt_labels.numpy())
        pred_bboxes.append(cur_pred_bboxes.numpy())
        pred_labels.append(cur_pred_labels.numpy())
        pred_scores.append(cur_pred_scores.numpy())

    return eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=CONFIG['evaluate_iou_threshold'],
        use_07_metric=use_07_metric)


if __name__ == '__main__':
    graph = tf.Graph()
    with graph.as_default():
        cur_base_model, cur_training_model = _get_default_base_model()
        cur_training_dataset = _get_training_dataset()
        cur_evaluation_dataset = _get_evaluating_dataset()
        cur_optimizer = _get_default_optimizer()
        train(cur_training_dataset, cur_evaluation_dataset, cur_base_model, cur_training_model, cur_optimizer)
