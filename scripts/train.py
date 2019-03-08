import os
import time
import argparse
import numpy as np
import tensorflow as tf

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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
tf_logging.set_verbosity(tf_logging.INFO)

CONFIG = None


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


def _get_default_vgg16_model(slim_ckpt_file_path=None):
    return Vgg16FasterRcnn(
        slim_ckpt_file_path=slim_ckpt_file_path,

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


def _get_default_optimizer(use_adam):
    lr = tf.train.exponential_decay(learning_rate=CONFIG['learning_rate_start'],
                                    global_step=tf.train.get_or_create_global_step(),
                                    decay_steps=CONFIG['learning_rate_decay_steps'],
                                    decay_rate=CONFIG['learning_rate_decay_rate'],
                                    staircase=True)
    if use_adam:
        return tf.train.AdamOptimizer(lr)
    else:
        return tf.train.MomentumOptimizer(lr, momentum=CONFIG['optimizer_momentum'])


def _get_training_dataset(preprocessing_type='caffe', dataset_type='pascal', data_root_path=None):
    if dataset_type == 'pascal':
        # 使用 trainaval 模式的 tfrecords 文件，共5个
        file_names = [os.path.join(data_root_path, 'pascal_trainval_%02d.tfrecords' % i) for i in range(5)]
        dataset = pascal_get_dataset(file_names,
                                     min_size=CONFIG['image_min_size'], max_size=CONFIG['image_max_size'],
                                     preprocessing_type=preprocessing_type,
                                     argument=True, shuffle=True)
    elif dataset_type == 'coco':
        dataset = coco_get_dataset(root_dir=data_root_path, mode='train',
                                   min_size=CONFIG['image_min_size'], max_size=CONFIG['image_max_size'],
                                   preprocessing_type=preprocessing_type,
                                   argument=False, )
    else:
        raise ValueError('unknown dataset type {}'.format(dataset_type))
    return dataset


def _get_rpn_l2_loss(base_model):
    l2_loss = 0
    for var in base_model.get_layer('vgg16').variables:
        if 'bias' in var.name or 'block1' in var.name or 'block2' in var.name:
            continue
        l2_loss = l2_loss + tf.reduce_sum(tf.square(var))
    for var in base_model.get_layer('rpn_head').variables:
        if 'bias' in var.name:
            continue
        l2_loss = l2_loss + tf.reduce_sum(tf.square(var))
    l2_loss = l2_loss * CONFIG['weight_decay']
    return l2_loss


def _get_roi_l2_loss(base_model):
    l2_loss = 0
    for var in base_model.get_layer('roi_head').variables:
        if 'bias' in var.name:
            continue
        l2_loss = l2_loss + tf.reduce_sum(tf.square(var))
    l2_loss = l2_loss * CONFIG['weight_decay']
    return l2_loss


def train_step_end2end(image, gt_bboxes, gt_labels,
                       base_model, optimizer, loss_type, ):
    with tf.GradientTape() as tape:
        rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss = base_model((image, gt_bboxes, gt_labels), True)

        if loss_type == 'total':
            l2_loss = tf.add_n(base_model.losses)
            total_loss = rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss + l2_loss
        elif loss_type == 'rpn':
            l2_loss = _get_rpn_l2_loss(base_model)
            total_loss = rpn_cls_loss + rpn_reg_loss + l2_loss
        else:
            l2_loss = _get_roi_l2_loss(base_model)
            total_loss = roi_cls_loss + roi_reg_loss + l2_loss

        train_step(base_model, total_loss, tape, optimizer)
        return rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss, l2_loss, total_loss


def train_one_epoch(dataset, base_model, optimizer, loss_type,
                    logging_every_n_steps,
                    summary_every_n_steps,
                    saver, save_every_n_steps, save_path):
    idx = 0

    for image, gt_bboxes, gt_labels in tqdm(dataset):
        # conver ymin xmin ymax xmax -> xmin ymin xmax ymax
        gt_bboxes = tf.squeeze(gt_bboxes, axis=0)
        channels = tf.split(gt_bboxes, 4, axis=1)
        gt_bboxes = tf.concat([
            channels[1], channels[0], channels[3], channels[2]
        ], axis=1)

        gt_labels = tf.to_int32(tf.squeeze(gt_labels, axis=0))

        rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss, l2_loss, total_loss = train_step_end2end(image,
                                                                                                         gt_bboxes,
                                                                                                         gt_labels,
                                                                                                         base_model,
                                                                                                         optimizer,
                                                                                                         loss_type, )

        # summary
        if idx % summary_every_n_steps == 0:
            summary.scalar("l2_loss", l2_loss)
            summary.scalar("rpn_cls_loss", rpn_cls_loss)
            summary.scalar("rpn_reg_loss", rpn_reg_loss)
            summary.scalar("roi_cls_loss", roi_cls_loss)
            summary.scalar("roi_reg_loss", roi_reg_loss)
            summary.scalar("total_loss", total_loss)

            pred_bboxes, pred_labels, pred_scores = base_model(image, False)

            if pred_bboxes is not None:
                selected_idx = tf.where(pred_scores >= CONFIG['show_image_score_threshold'])[:, 0]
                if tf.size(selected_idx) != 0:
                    # gt
                    gt_channels = tf.split(gt_bboxes, 4, axis=1)
                    show_gt_bboxes = tf.concat([gt_channels[1], gt_channels[0], gt_channels[3], gt_channels[2]], axis=1)
                    gt_image = show_one_image(tf.squeeze(image, axis=0).numpy(), show_gt_bboxes.numpy(),
                                              gt_labels.numpy(), enable_matplotlib=False)
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
                                                pred_labels.numpy(), enable_matplotlib=False)
                    tf.contrib.summary.image("pred_image", tf.expand_dims(pred_image, axis=0))

        # logging
        if idx % logging_every_n_steps == 0:
            if isinstance(optimizer, tf.train.AdamOptimizer):
                show_lr = optimizer._lr()
            else:
                show_lr = optimizer._learning_rate()
            tf_logging.info('steps %d, lr is %.5f, '
                            'loss: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (idx + 1, show_lr,
                                                                          rpn_cls_loss, rpn_reg_loss,
                                                                          roi_cls_loss, roi_reg_loss,
                                                                          l2_loss, total_loss)
                            )

        # saving
        if saver is not None and save_path is not None and idx % save_every_n_steps == 0 and idx != 0:
            saver.save(os.path.join(save_path, 'model.ckpt'), global_step=tf.train.get_or_create_global_step())

        idx += 1


def train(training_dataset, base_model, optimizer, loss_type,
          logging_every_n_steps,
          save_every_n_steps,
          summary_every_n_steps,
          train_dir,
          ckpt_dir,
          restore_ckpt_file_path,
          tf_faster_rcnn_ckpt_file_path,
          ):

    # 重大bug……
    # 如果没有进行这步操作，keras模型中rpn head的参数并没有初始化，不存在于后续 base_model.variables 中
    base_model(tf.to_float(np.random.rand(1, 800, 600, 3), False))

    # 获取 pretrained model
    saver = eager_saver.Saver(base_model.variables)

    # 命令行指定 ckpt file
    if restore_ckpt_file_path is not None:
        saver.restore(restore_ckpt_file_path)

    # tf_faster_rcnn 预训练模型
    if tf_faster_rcnn_ckpt_file_path is not None:
        base_model.load_tf_faster_rcnn_tf_weights(tf_faster_rcnn_ckpt_file_path)

    # 当前 logs_dir 中的预训练模型，用于继续训练
    if tf.train.latest_checkpoint(ckpt_dir) is not None:
        saver.restore(tf.train.latest_checkpoint(ckpt_dir))

    tf.train.get_or_create_global_step()
    train_writer = tf.contrib.summary.create_file_writer(train_dir, flush_millis=100000)
    for i in range(CONFIG['epochs']):
        tf_logging.info('epoch %d starting...' % (i + 1))
        start = time.time()
        with train_writer.as_default(), summary.always_record_summaries():
            train_one_epoch(training_dataset, base_model, optimizer, loss_type,
                            logging_every_n_steps=logging_every_n_steps,
                            summary_every_n_steps=summary_every_n_steps,
                            saver=saver, save_every_n_steps=save_every_n_steps, save_path=ckpt_dir,
                            )
        train_end = time.time()
        tf_logging.info('epoch %d training finished, costing %d seconds...' % (i + 1, train_end - start))


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN model')
    parser.add_argument('--data_type', default="pascal", type=str, help='pascal or coco')
    parser.add_argument('--logging_every_n_steps', default=100, type=int)
    parser.add_argument('--saving_every_n_steps', default=5000, type=int)
    parser.add_argument('--summary_every_n_steps', default=100, type=int)
    parser.add_argument('--restore_ckpt_path', type=str, default=None)
    parser.add_argument('--use_adam', type=bool, default=False)
    parser.add_argument('--loss_type', type=str, default='total', help='one of [total, rpn, roi]')

    # parser.add_argument('--data_root_path', default="/ssd/zhangyiyang/COCO2017", type=str)
    parser.add_argument('--data_root_path', type=str,
                        default="/ssd/zhangyiyang/tf_eager_object_detection/VOCdevkit/tf_eager_records")
    parser.add_argument('--logs_dir', type=str,
                        default="/ssd/zhangyiyang/test/tf_eager_object_detection/logs/logs-pascal-slim-adam-1e-3")
    parser.add_argument('--slim_ckpt_file_path', type=str,
                        default="/ssd/zhangyiyang/slim/vgg_16.ckpt")
    parser.add_argument('--tf_faster_rcnn_ckpt_file_path', type=str, default=None)
    # parser.add_argument('--tf_faster_rcnn_ckpt_file_path', type=str,
    #                     default='/ssd/zhangyiyang/tf_eager_object_detection/voc_2007_trainval/'
    #                             'vgg16_faster_rcnn_iter_70000.ckpt')

    # parser.add_argument('--data_root_path', default="D:\\data\\COCO2017", type=str)
    # parser.add_argument('--data_root_path', default="D:\\data\\VOCdevkit\\tf_eager_records\\", type=str)
    # parser.add_argument('--logs_dir', default="D:\\data\\logs\\logs-pascal", type=str)

    args = parser.parse_args()
    return args


def main(args):
    global CONFIG
    if args.data_type == 'coco':
        CONFIG = COCO_CONFIG
    elif args.data_type == 'pascal':
        CONFIG = PASCAL_CONFIG
    else:
        raise ValueError('unknown data_type {}'.format(args.data_type))

    if args.loss_type not in ['total', 'roi', 'rpn']:
        raise ValueError('unknown loss type {}'.format(args.loss_type))

    train(training_dataset=_get_training_dataset('caffe', args.data_type, args.data_root_path),
          base_model=_get_default_vgg16_model(slim_ckpt_file_path=args.slim_ckpt_file_path),
          optimizer=_get_default_optimizer(args.use_adam),
          loss_type=args.loss_type,

          logging_every_n_steps=args.logging_every_n_steps,
          save_every_n_steps=args.saving_every_n_steps,
          summary_every_n_steps=args.summary_every_n_steps,

          train_dir=os.path.join(args.logs_dir, 'train'),
          ckpt_dir=os.path.join(args.logs_dir, 'ckpt'),
          restore_ckpt_file_path=args.restore_ckpt_path,
          tf_faster_rcnn_ckpt_file_path=args.tf_faster_rcnn_ckpt_file_path,
          )


if __name__ == '__main__':
    main(parse_args())
