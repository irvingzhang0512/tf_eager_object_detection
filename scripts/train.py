import tensorflow as tf
import os
import time
import matplotlib

matplotlib.use('agg')

from object_detection.model.vgg16_faster_rcnn import Vgg16FasterRcnn
from object_detection.config.faster_rcnn_config import COCO_CONFIG, PASCAL_CONFIG
from object_detection.utils.visual_utils import show_one_image
from object_detection.dataset.pascal_tf_dataset_generator import get_dataset as pascal_get_dataset
from object_detection.dataset.coco_tf_dataset_generator import get_dataset as coco_get_dataset
from tensorflow.contrib.summary import summary
from tensorflow.contrib.eager.python import saver as eager_saver
from tqdm import tqdm
from tensorflow.python.platform import tf_logging

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.enable_eager_execution()
tf_logging.set_verbosity(tf_logging.INFO)

DATASET_TYPE = 'pascal'
if DATASET_TYPE == 'pascal':
    CONFIG = PASCAL_CONFIG
elif DATASET_TYPE == 'coco':
    CONFIG = COCO_CONFIG
else:
    raise ValueError('Unknown Dataset Type')

# train_records_list = [
#     '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_trainval_00.tfrecords',
#     '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_trainval_01.tfrecords',
#     '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_trainval_02.tfrecords',
#     '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_trainval_03.tfrecords',
#     '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_trainval_04.tfrecords',
# ]
# eval_records_list = [
#     '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_test_00.tfrecords',
#     '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_test_01.tfrecords',
#     '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_test_02.tfrecords',
#     '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_test_03.tfrecords',
#     '/home/tensorflow05/data/VOCdevkit/tf_eager_records/pascal_test_04.tfrecords',
# ]
# cur_train_dir = '/home/tensorflow05/zyy/tf_eager_object_detection/logs/logs-pascal/train'
# cur_ckpt_dir = '/home/tensorflow05/zyy/tf_eager_object_detection/logs/logs-pascal/ckpt'
# coco_root_path = "/home/tensorflow05/data/COCO2017"


# train_records_list = [
#     '/ssd/zhangyiyang/tf_eager_object_detection/VOCdevkit/tf_eager_records/pascal_trainval_00.tfrecords',
#     '/ssd/zhangyiyang/tf_eager_object_detection/VOCdevkit/tf_eager_records/pascal_trainval_01.tfrecords',
#     '/ssd/zhangyiyang/tf_eager_object_detection/VOCdevkit/tf_eager_records/pascal_trainval_02.tfrecords',
#     '/ssd/zhangyiyang/tf_eager_object_detection/VOCdevkit/tf_eager_records/pascal_trainval_03.tfrecords',
#     '/ssd/zhangyiyang/tf_eager_object_detection/VOCdevkit/tf_eager_records/pascal_trainval_04.tfrecords',
# ]
# eval_records_list = [
#     '/ssd/zhangyiyang/tf_eager_object_detection/VOCdevkit/tf_eager_records/pascal_test_00.tfrecords',
#     '/ssd/zhangyiyang/tf_eager_object_detection/VOCdevkit/tf_eager_records/pascal_test_01.tfrecords',
#     '/ssd/zhangyiyang/tf_eager_object_detection/VOCdevkit/tf_eager_records/pascal_test_02.tfrecords',
#     '/ssd/zhangyiyang/tf_eager_object_detection/VOCdevkit/tf_eager_records/pascal_test_03.tfrecords',
#     '/ssd/zhangyiyang/tf_eager_object_detection/VOCdevkit/tf_eager_records/pascal_test_04.tfrecords',
# ]
# cur_train_dir = '/ssd/zhangyiyang/tf_eager_object_detection/logs-pascal-1'
# cur_ckpt_dir = '/ssd/zhangyiyang/tf_eager_object_detection/logs-pascal-1'
# coco_root_path = "/ssd/zhangyiyang/COCO2017"


train_records_list = [
    'D:\\data\\VOCdevkit\\tf_eager_records\\pascal_trainval_00.tfrecords',
]
eval_records_list = [
    'D:\\data\\VOCdevkit\\tf_eager_records\\pascal_test_00.tfrecords',
]
cur_train_dir = 'D:\\data\\logs\\logs-pascal\\train'
cur_ckpt_dir = 'D:\\data\\logs\\logs-pascal\\ckpt'
coco_root_path = 'D:\\data\\COCO2017'


def apply_gradients(model, optimizer, gradients):
    all_vars = model.variables
    if CONFIG['learning_rate_bias_double']:
        all_grads = []
        all_vars = []
        for grad, var in zip(gradients, model.variables):
            if grad is None:
                continue
            scale = 1.0
            if 'biases' in var.name:
                scale = 2.0
            all_grads.append(grad * scale)
            all_vars.append(var)
        gradients = all_grads
    optimizer.apply_gradients(zip(gradients, all_vars),
                              global_step=tf.train.get_or_create_global_step())


def compute_gradients(model, loss, tape):
    return tape.gradient(loss, model.variables)


def train_step(model, loss, tape, optimizer):
    apply_gradients(model, optimizer, compute_gradients(model, loss, tape))


def _get_default_vgg16_model():
    return Vgg16FasterRcnn(
        num_classes=CONFIG['num_classes'],
        weight_decay=CONFIG['weight_decay'],

        ratios=CONFIG['ratios'],
        scales=CONFIG['scales'],
        extractor_stride=CONFIG['extractor_stride'],

        rpn_proposal_means=CONFIG['rpn_proposal_means'],
        rpn_proposal_stds=CONFIG['rpn_proposal_stds'],

        rpn_proposal_num_pre_nms_train=CONFIG['rpn_proposal_train_pre_nms_sample_number'],
        rpn_proposal_num_post_nms_train=CONFIG['rpn_proposal_train_after_nms_sample_number'],
        rpn_proposal_num_pre_nms_test=CONFIG['rpn_proposal_test_pre_nms_sample_number'],
        rpn_proposal_num_post_nms_test=CONFIG['rpn_proposal_test_after_nms_sample_number'],
        rpn_proposal_nms_iou_threshold=CONFIG['rpn_proposal_nms_iou_threshold'],

        rpn_sigma=CONFIG['rpn_sigma'],
        rpn_training_pos_iou_threshold=CONFIG['rpn_pos_iou_threshold'],
        rpn_training_neg_iou_threshold=CONFIG['rpn_neg_iou_threshold'],
        rpn_training_total_num_samples=CONFIG['rpn_total_sample_number'],
        rpn_training_max_pos_samples=CONFIG['rpn_pos_sample_max_number'],

        roi_proposal_means=CONFIG['roi_proposal_means'],
        roi_proposal_stds=CONFIG['roi_proposal_stds'],

        roi_pool_size=CONFIG['roi_pooling_size'],
        roi_head_keep_dropout_rate=CONFIG['roi_head_keep_dropout_rate'],
        roi_feature_size=CONFIG['roi_feature_size'],

        roi_sigma=CONFIG['roi_sigma'],
        roi_training_pos_iou_threshold=CONFIG['roi_pos_iou_threshold'],
        roi_training_neg_iou_threshold=CONFIG['roi_neg_iou_threshold'],
        roi_training_total_num_samples=CONFIG['roi_total_sample_number'],
        roi_training_max_pos_samples=CONFIG['roi_pos_sample_max_number'],

        prediction_max_objects_per_image=CONFIG['max_objects_per_image'],
        prediction_max_objects_per_class=CONFIG['max_objects_per_class_per_image'],
        prediction_nms_iou_threshold=CONFIG['predictions_nms_iou_threshold'],
        prediction_score_threshold=CONFIG['prediction_score_threshold'],
    )


def _get_default_optimizer():
    lr = tf.train.exponential_decay(learning_rate=CONFIG['learning_rate_start'],
                                    global_step=tf.train.get_or_create_global_step(),
                                    decay_steps=CONFIG['learning_rate_decay_steps'],
                                    decay_rate=CONFIG['learning_rate_decay_rate'],
                                    staircase=True)
    return tf.train.MomentumOptimizer(lr, momentum=CONFIG['optimizer_momentum'])


def _get_training_dataset(preprocessing_type='caffe', dataset_type='pascal'):
    if dataset_type == 'pascal':
        dataset = pascal_get_dataset(train_records_list,
                                     min_size=CONFIG['image_min_size'], max_size=CONFIG['image_max_size'],
                                     preprocessing_type=preprocessing_type,
                                     argument=False, )
    elif dataset_type == 'coco':
        dataset = coco_get_dataset(root_dir=coco_root_path, mode='train',
                                   min_size=CONFIG['image_min_size'], max_size=CONFIG['image_max_size'],
                                   preprocessing_type=preprocessing_type,
                                   argument=False, )
    else:
        raise ValueError('unknown dataset type {}'.format(dataset_type))
    return dataset


class MeanOps:
    def __init__(self):
        self.total = .0
        self.cnt = 0

    def mean(self):
        if self.cnt == 0:
            return None
        return self.total / self.cnt

    def update(self, cur):
        self.total += cur
        self.cnt += 1

    def reset(self):
        self.total = .0
        self.cnt = 0


def train_one_epoch(dataset, base_model, optimizer,
                    logging_every_n_steps=20,
                    summary_every_n_steps=50,
                    saver=None, save_every_n_steps=2500, save_path=None,
                    show_score_threshold=0.3):
    idx = 0

    rpn_cls_mean = MeanOps()
    rpn_reg_mean = MeanOps()
    roi_cls_mean = MeanOps()
    roi_reg_mean = MeanOps()
    total_mean = MeanOps()

    for image, gt_bboxes, gt_labels, _ in tqdm(dataset):
        # conver ymin xmin ymax xmax -> xmin ymin xmax ymax
        gt_bboxes = tf.squeeze(gt_bboxes, axis=0)
        channels = tf.split(gt_bboxes, 4, axis=1)
        gt_bboxes = tf.concat([
            channels[1], channels[0], channels[3], channels[2]
        ], axis=1)

        gt_labels = tf.to_int32(tf.squeeze(gt_labels, axis=0))

        with tf.GradientTape() as tape:
            rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss = base_model((image, gt_bboxes, gt_labels), True)
            l2_loss = tf.add_n(base_model.losses)
            total_loss = rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss + l2_loss

            train_step(base_model, total_loss, tape, optimizer)

        if idx % summary_every_n_steps == 0:
            rpn_cls_mean.update(rpn_cls_loss)
            rpn_reg_mean.update(rpn_reg_loss)
            roi_cls_mean.update(roi_cls_loss)
            roi_reg_mean.update(roi_reg_loss)
            total_mean.update(total_loss)
            summary.scalar("rpn_cls_loss", rpn_cls_mean.mean())
            summary.scalar("rpn_reg_loss", rpn_reg_mean.mean())
            summary.scalar("roi_cls_loss", roi_cls_mean.mean())
            summary.scalar("roi_reg_loss", roi_reg_mean.mean())
            summary.scalar("l2_loss", l2_loss)
            summary.scalar("total_loss", total_mean.mean())

        if idx % logging_every_n_steps == 0:
            tf_logging.info('steps %d loss: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (idx + 1,
                                                                                   rpn_cls_loss, rpn_reg_loss,
                                                                                   roi_cls_loss, roi_reg_loss,
                                                                                   tf.add_n(base_model.losses),
                                                                                   total_loss)
                            )
            pred_bboxes, pred_labels, pred_scores = base_model(image, False)

            if pred_bboxes is not None:
                selected_idx = tf.where(pred_scores >= show_score_threshold)[:, 0]
                if tf.size(selected_idx) != 0:
                    # gt
                    gt_channels = tf.split(gt_bboxes, 4, axis=1)
                    show_gt_bboxes = tf.concat([gt_channels[1], gt_channels[0], gt_channels[3], gt_channels[2]], axis=1)
                    gt_image = show_one_image(tf.squeeze(image, axis=0).numpy(), show_gt_bboxes.numpy(),
                                              gt_labels.numpy())
                    tf.contrib.summary.image("gt_image", tf.expand_dims(gt_image, axis=0))

                    # pred
                    pred_bboxes = tf.gather(pred_bboxes, selected_idx)
                    pred_labels = tf.gather(pred_labels, selected_idx)
                    channels = tf.split(pred_bboxes, num_or_size_splits=4, axis=1)
                    show_pred_bboxes = tf.concat([
                        channels[1], channels[0], channels[3], channels[2]
                    ], axis=1)
                    pred_image = show_one_image(tf.squeeze(image, axis=0).numpy(),
                                                show_pred_bboxes.numpy(),
                                                pred_labels.numpy())
                    tf.contrib.summary.image("pred_image", tf.expand_dims(pred_image, axis=0))

        # if saver is not None and save_path is not None and idx % save_every_n_steps == 0:
        if saver is not None and save_path is not None and idx % save_every_n_steps == 0:
            saver.save(os.path.join(save_path, 'model.ckpt'), global_step=tf.train.get_or_create_global_step())

        idx += 1


def train(training_dataset, base_model, optimizer,
          logging_every_n_steps=100,
          save_every_n_steps=5000,
          summary_every_n_steps=10,

          train_dir=cur_train_dir,
          ckpt_dir=cur_ckpt_dir,
          ):
    saver = eager_saver.Saver(base_model.variables)

    # load saved ckpt files
    if tf.train.latest_checkpoint(ckpt_dir) is not None:
        saver.restore(tf.train.latest_checkpoint(ckpt_dir))
        tf_logging.info('restore from {}...'.format(tf.train.latest_checkpoint(ckpt_dir)))

    tf.train.get_or_create_global_step()
    train_writer = tf.contrib.summary.create_file_writer(train_dir, flush_millis=100000)
    for i in range(CONFIG['epochs']):
        tf_logging.info('epoch %d starting...' % (i + 1))
        start = time.time()
        with train_writer.as_default(), summary.always_record_summaries():
            train_one_epoch(training_dataset, base_model, optimizer,
                            logging_every_n_steps=logging_every_n_steps,
                            summary_every_n_steps=summary_every_n_steps,
                            saver=saver,
                            save_every_n_steps=save_every_n_steps,
                            save_path=ckpt_dir
                            )
        train_end = time.time()
        tf_logging.info('epoch %d training finished, costing %d seconds, start evaluating...' % (i + 1,
                                                                                                 train_end - start))


if __name__ == '__main__':

    cur_model = _get_default_vgg16_model()
    cur_training_dataset = _get_training_dataset('caffe', DATASET_TYPE)

    cur_optimizer = _get_default_optimizer()
    train(cur_training_dataset, cur_model, cur_optimizer)
