import matplotlib

matplotlib.use('agg')

import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from object_detection.utils.evaluation.pascal_eval_utils import get_prediction_files
from object_detection.utils.evaluation.detectron_pascal_ap_utils import voc_eval
from object_detection.model.vgg16_faster_rcnn import Vgg16FasterRcnn
from object_detection.config.faster_rcnn_config import PASCAL_CONFIG as CONFIG
from tensorflow.contrib.eager.python import saver as eager_saver

num_classes = 21,
class_list = ('__background__',  # always index 0
              'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor')


def eval_from_scratch(model,
                      dataset_type='tf',
                      root_path='/home/tensorflow05/data/VOCdevkit/VOC2007',
                      result_file_format='/home/tensorflow05/zyy/tf_eager_object_detection/results/{:s}.txt',
                      cache_dir='/home/tensorflow05/zyy/tf_eager_object_detection/results',

                      prediction_score_threshold=CONFIG['prediction_score_threshold'],
                      iou_threshold=CONFIG['predictions_nms_iou_threshold'],
                      max_objects_per_class=CONFIG['max_objects_per_class_per_image'],
                      max_objects_per_image=CONFIG['max_objects_per_image'],
                      target_means=CONFIG['roi_proposal_means'],
                      target_stds=CONFIG['roi_proposal_stds'],
                      evaluation_iou_threshold=CONFIG['evaluate_iou_threshold']
                      ):
    """

    :param model:                       导入好参数的模型
    :param dataset_type:                训练时使用的原始数据是通过 cv2 产生还是 tf 产生
    :param root_path:                   VOC的目录，要具体到某一年
    :param result_file_format:          要将test结果写到文件中，文件路径为 result_file_format.format(class_name)
    :param prediction_score_threshold:  预测时的 score threshold
    :param iou_threshold:               获取预测结果时 nms 的 iou threshold
    :param max_objects_per_class:       每张图片中，每一类物体的最大数量
    :param max_objects_per_image:       每张图片中，所有物体的最大数量
    :param target_means:                bbox_txtytwth 转换时的参数
    :param target_stds:                 bbox_txtytwth 转换时的参数
    :param cache_dir:                   预测时，会将gt的信息使用pickle进行保存，保存的路径就是 cache_dir+'test_annots.pkl'
    :param evaluation_iou_threshold:    判断预测结果与gt的 iou 大于该值时，认为是真实值
    :return:
    """
    # 生成检测结果的本地文件
    get_prediction_files(model,
                         dataset_type=dataset_type,
                         data_root_path=root_path,
                         mode='test',
                         result_file_format=result_file_format,
                         score_threshold=prediction_score_threshold, iou_threshold=iou_threshold,
                         max_objects_per_class=max_objects_per_class, max_objects_per_image=max_objects_per_image,
                         target_means=target_means, target_stds=target_stds)

    # 通过本地文件（包括检测结果和真实结果）计算map
    eval_by_local_files_and_gt_xmls(root_path,
                                    result_file_format,
                                    cache_dir,
                                    evaluation_iou_threshold)


def eval_by_local_files_and_gt_xmls(root_path,
                                    result_file_format,
                                    cache_dir,
                                    prediction_iou_threshold=CONFIG['evaluate_iou_threshold'], ):
    annotation_file_format = os.path.join(root_path, 'Annotations', "{:s}.xml")
    imagesetfile = os.path.join(root_path, 'ImageSets', 'Main', 'test.txt')
    all_ap = .0
    for cls_name in class_list:
        if cls_name == '__background__':
            continue
        cur_res = voc_eval(result_file_format,
                           annotation_file_format,
                           imagesetfile,
                           cls_name,
                           cache_dir,
                           ovthresh=prediction_iou_threshold,
                           use_07_metric=True,
                           )
        print('class {} get ap {}'.format(cls_name, cur_res[2]))
        all_ap += cur_res[2]
    print('map {}'.format(all_ap / (len(class_list) - 1)))


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


def _get_tf_faster_rcnn_pre_trained_model(model,
                                          ckpt_file_path='/home/tensorflow05/data/voc_2007_trainval/'
                                                         'vgg16_faster_rcnn_iter_70000.ckpt'):
    model(
        (
            tf.constant(np.random.rand(1, 600, 800, 3), dtype=tf.float32),
            tf.to_int32(np.random.rand(2, 4)),
            tf.to_int32(np.array([1, 10]))
        ), True)
    model.load_tf_faster_rcnn_tf_weights(ckpt_file_path)
    return model


def _load_from_ckpt_file(model, ckpt_file_path):
    saver = eager_saver.Saver(model.variables)
    if tf.train.latest_checkpoint(ckpt_file_path) is not None:
        saver.restore(tf.train.latest_checkpoint(ckpt_file_path))
        tf.logging.info('restore from {}...'.format(tf.train.latest_checkpoint(ckpt_file_path)))


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Evaluate a Fast R-CNN model')
    parser.add_argument('ckpt_file_path', type=str, help='target ckpt file path', )
    parser.add_argument('--use_tf_faster_rcnn_model', default=True, type=bool)
    parser.add_argument('--use_local_result_files', default=False, type=bool)
    parser.add_argument('--dataset_type', help='type of dataset, cv2 or tf',
                        default='tf', type=str)
    parser.add_argument('--root_path', help='path to pascal voc 2007',
                        default='/home/tensorflow05/data/VOCdevkit/VOC2007', type=str)
    parser.add_argument('--result_file_format', help='local detection result file pattern',
                        default='/home/tensorflow05/zyy/tf_eager_object_detection/results/{:s}.txt', type=str)
    parser.add_argument('--annotation_cache_dir', help='path to save annotation cache pickle file',
                        default='/home/tensorflow05/zyy/tf_eager_object_detection/results', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def main(args):
    if args.use_local_result_files:
        eval_by_local_files_and_gt_xmls(root_path=args.root_path,
                                        result_file_format=args.result_file_format,
                                        cache_dir=args.annotation_cache_dir,
                                        )
        return

    cur_model = _get_default_vgg16_model()
    if args.use_tf_faster_rcnn_model:
        cur_model = _get_tf_faster_rcnn_pre_trained_model(cur_model, args.ckpt_file_path)
    else:
        _load_from_ckpt_file(cur_model, args.ckpt_file_path)
    eval_from_scratch(cur_model,
                      dataset_type=args.dataset_type,
                      root_path=args.root_path,
                      result_file_format=args.result_file_format,
                      cache_dir=args.annotation_cache_dir)


if __name__ == '__main__':
    main(parse_args())
