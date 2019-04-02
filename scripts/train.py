import os
import time
import argparse
import numpy as np
import tensorflow as tf

from object_detection.model.model_factory import model_factory
from object_detection.config.config_factory import config_factory
from object_detection.utils.visual_utils import show_one_image
from object_detection.dataset.pascal_tf_dataset_generator import get_dataset as pascal_get_dataset
from object_detection.dataset.coco_tf_dataset_generator import get_dataset as coco_get_dataset
from tensorflow.contrib.summary import summary
from tensorflow.contrib.eager.python import saver as eager_saver
from tensorflow.python.platform import tf_logging
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
tf_logging.set_verbosity(tf_logging.INFO)

CONFIG = None


def train_step(model, loss, tape, optimizer):
    all_vars = model.variables
    gradients = tape.gradient(loss, all_vars)

    if CONFIG['learning_rate_bias_double']:
        all_grads = []
        all_vars = []
        for grad, var in zip(gradients, model.variables):
            if grad is None:
                continue
            scale = 1.0
            if 'bias' in var.name:
                scale = 2.0
            all_grads.append(grad * scale)
            all_vars.append(var)
        gradients = all_grads

    optimizer.apply_gradients(zip(gradients, all_vars),
                              global_step=tf.train.get_or_create_global_step())


def _get_default_optimizer(use_adam):
    lr = tf.train.piecewise_constant(tf.train.get_or_create_global_step(),
                                     boundaries=CONFIG['learning_rate_multi_decay_steps'],
                                     values=CONFIG['learning_rate_multi_lrs'])
    if use_adam:
        return tf.train.AdamOptimizer(lr)
    else:
        return tf.train.MomentumOptimizer(lr, momentum=CONFIG['optimizer_momentum'])


def _get_training_dataset(preprocessing_type='caffe', dataset_type='pascal',
                          pascal_year="2007", pascal_mode='trainval', pascal_tf_records_num=5,
                          data_root_path=None):
    if dataset_type == 'pascal':
        base_pattern = 'pascal_{}_{}_%02d.tfrecords'.format(pascal_year, pascal_mode)
        file_names = [os.path.join(data_root_path, base_pattern % i) for i in range(pascal_tf_records_num)]
        dataset = pascal_get_dataset(file_names,
                                     min_size=CONFIG['image_min_size'], max_size=CONFIG['image_max_size'],
                                     preprocessing_type=preprocessing_type, caffe_pixel_means=CONFIG['bgr_pixel_means'],
                                     argument=True)
    elif dataset_type == 'coco':
        dataset = coco_get_dataset(root_dir=data_root_path, mode='train',
                                   min_size=CONFIG['image_min_size'], max_size=CONFIG['image_max_size'],
                                   preprocessing_type=preprocessing_type, caffe_pixel_means=CONFIG['bgr_pixel_means'],
                                   argument=True, )
    else:
        raise ValueError('unknown dataset type {}'.format(dataset_type))
    return dataset


def train_one_epoch(dataset, base_model, optimizer,
                    preprocessing_type,
                    logging_every_n_steps,
                    summary_every_n_steps,
                    saver, save_every_n_steps, save_path):
    idx = 0

    for image, gt_bboxes, gt_labels in tqdm(dataset):
        # bgr input
        # for keras application pre-trained models, use bgr

        # conver ymin xmin ymax xmax -> xmin ymin xmax ymax
        gt_bboxes = tf.squeeze(gt_bboxes, axis=0)
        channels = tf.split(gt_bboxes, 4, axis=1)
        gt_bboxes = tf.concat([
            channels[1], channels[0], channels[3], channels[2]
        ], axis=1)

        # set labels to int32
        gt_labels = tf.to_int32(tf.squeeze(gt_labels, axis=0))

        # train one step
        with tf.GradientTape() as tape:
            rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss = base_model((image, gt_bboxes, gt_labels), True)
            l2_loss = tf.add_n(base_model.losses)
            total_loss = rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss + l2_loss
            train_step(base_model, total_loss, tape, optimizer)

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
                    # show gt
                    gt_channels = tf.split(gt_bboxes, 4, axis=1)
                    show_gt_bboxes = tf.concat([gt_channels[1], gt_channels[0], gt_channels[3], gt_channels[2]], axis=1)
                    gt_image = show_one_image(tf.squeeze(image, axis=0).numpy(), show_gt_bboxes.numpy(),
                                              gt_labels.numpy(),
                                              preprocessing_type=preprocessing_type,
                                              caffe_pixel_means=CONFIG['bgr_pixel_means'],
                                              enable_matplotlib=False)
                    tf.contrib.summary.image("gt_image", tf.expand_dims(gt_image, axis=0))

                    # show pred
                    pred_bboxes = tf.gather(pred_bboxes, selected_idx)
                    pred_labels = tf.gather(pred_labels, selected_idx)
                    channels = tf.split(pred_bboxes, num_or_size_splits=4, axis=1)
                    show_pred_bboxes = tf.concat([
                        channels[1], channels[0], channels[3], channels[2]
                    ], axis=1)
                    pred_image = show_one_image(tf.squeeze(image, axis=0).numpy(),
                                                show_pred_bboxes.numpy(),
                                                pred_labels.numpy(),
                                                preprocessing_type=preprocessing_type,
                                                caffe_pixel_means=CONFIG['bgr_pixel_means'],
                                                enable_matplotlib=False)
                    tf.contrib.summary.image("pred_image", tf.expand_dims(pred_image, axis=0))

        # logging
        if idx % logging_every_n_steps == 0:
            if isinstance(optimizer, tf.train.AdamOptimizer):
                show_lr = optimizer._lr()
            else:
                show_lr = optimizer._learning_rate()
            logging_format = 'steps %d, lr is %.5f, loss: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f'
            tf_logging.info(logging_format % (idx + 1, show_lr,
                                              rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss,
                                              l2_loss, total_loss))

        # saving
        if saver is not None and save_path is not None and idx % save_every_n_steps == 0 and idx != 0:
            saver.save(os.path.join(save_path, 'model.ckpt'), global_step=tf.train.get_or_create_global_step())

        idx += 1


def train(training_dataset,
          preprocessing_type,

          base_model,

          optimizer,

          logging_every_n_steps,
          save_every_n_steps,
          summary_every_n_steps,

          train_dir,
          ckpt_dir,
          restore_ckpt_file_path,
          ):
    # 获取 pretrained model
    saver = eager_saver.Saver(base_model.variables)

    # 命令行指定 ckpt file
    if restore_ckpt_file_path is not None:
        saver.restore(restore_ckpt_file_path)

    # 当前 logs_dir 中的预训练模型，用于继续训练
    print(ckpt_dir)
    if tf.train.latest_checkpoint(ckpt_dir) is not None:
        saver.restore(tf.train.latest_checkpoint(ckpt_dir))

    tf.train.get_or_create_global_step()
    train_writer = tf.contrib.summary.create_file_writer(train_dir, flush_millis=100000)
    for i in range(CONFIG['epochs']):
        tf_logging.info('epoch %d starting...' % (i + 1))
        start = time.time()
        with train_writer.as_default(), summary.always_record_summaries():
            train_one_epoch(dataset=training_dataset, base_model=base_model,
                            optimizer=optimizer, preprocessing_type=preprocessing_type,
                            logging_every_n_steps=logging_every_n_steps,
                            summary_every_n_steps=summary_every_n_steps,
                            saver=saver, save_every_n_steps=save_every_n_steps, save_path=ckpt_dir,
                            )
        tf.set_random_seed(1)
        train_end = time.time()
        tf_logging.info('epoch %d training finished, costing %d seconds...' % (i + 1, train_end - start))


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--gpu_id', default="0", type=str, help='used in sys variable CUDA_VISIBLE_DEVICES')

    parser.add_argument('--model_type', type=str, default='faster_rcnn',
                        help='one of [faster_rcnn, fpn]')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='one of [vgg16, resnet50, resnet101, resnet152]')

    parser.add_argument('--data_type', default="pascal", type=str, help='pascal or coco')
    parser.add_argument('--pascal_year', default="2007", type=str, help='one of [2007, 2012, 0712]')
    parser.add_argument('--pascal_mode', default="trainval", type=str, help='one of [trainval, train, val]')
    parser.add_argument('--pascal_tf_records_num', default=5, type=int, help='number of pascal tf records')

    parser.add_argument('--logging_every_n_steps', default=100, type=int)
    parser.add_argument('--saving_every_n_steps', default=5000, type=int)
    parser.add_argument('--summary_every_n_steps', default=100, type=int)
    parser.add_argument('--restore_ckpt_path', type=str, default=None)

    parser.add_argument('--use_adam', type=bool, default=False)

    parser.add_argument('--logs_name', type=str, default='default',
                        help='logs dir name pattern is `logs-{data_type}-{model_type}-{backbone}-{logs_name}`', )

    # parser.add_argument('--data_root_path', default="/ssd/zhangyiyang/COCO2017", type=str)
    parser.add_argument('--data_root_path', type=str,
                        help='path to tfrecord files if pascal, path to coco root if coco',
                        default="/ssd/zhangyiyang/tf_eager_object_detection/VOCdevkit/tf_eager_records")
    parser.add_argument('--logs_dir', type=str, help='path to save ckpt files and tensorboard summaries.',
                        default="/ssd/zhangyiyang/tf_eager_object_detection/logs")

    # # parser.add_argument('--data_root_path', default="D:\\data\\COCO2017", type=str)
    # parser.add_argument('--data_root_path', default="D:\\data\\VOCdevkit\\tf_eager_records\\", type=str)
    # parser.add_argument('--logs_dir', default="D:\\data\\logs\\logs-pascal", type=str)

    args = parser.parse_args()
    return args


def main(args):
    global CONFIG
    CONFIG = config_factory(args.data_type, args.model_type)

    # tensorflow eager 模式基本参数设置
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    tf.enable_eager_execution(config=config)

    # 建立模型，并初始化
    cur_model = model_factory(args.model_type, args.backbone, CONFIG)
    preprocessing_type = 'caffe'
    cur_model(tf.to_float(np.random.rand(1, 800, 600, 3)), False)

    # logs基本信息
    # logs-{data_type}-{model_type}-{backbone}-{logs_name}
    logs_name_pattern = 'logs-{}-{}-{}-{}'
    logs_path_name = logs_name_pattern.format(args.data_type, args.model_type, args.backbone, args.logs_name)

    # 开始训练
    train(training_dataset=_get_training_dataset(preprocessing_type=preprocessing_type,
                                                 dataset_type=args.data_type,
                                                 pascal_year=args.pascal_year,
                                                 pascal_mode=args.pascal_mode,
                                                 pascal_tf_records_num=args.pascal_tf_records_num,
                                                 data_root_path=args.data_root_path),
          preprocessing_type=preprocessing_type,

          base_model=cur_model,

          optimizer=_get_default_optimizer(args.use_adam),

          logging_every_n_steps=args.logging_every_n_steps,
          save_every_n_steps=args.saving_every_n_steps,
          summary_every_n_steps=args.summary_every_n_steps,

          train_dir=os.path.join(args.logs_dir, logs_path_name, 'train'),
          ckpt_dir=os.path.join(args.logs_dir, logs_path_name, 'ckpt'),
          restore_ckpt_file_path=args.restore_ckpt_path,
          )


if __name__ == '__main__':
    main(parse_args())
