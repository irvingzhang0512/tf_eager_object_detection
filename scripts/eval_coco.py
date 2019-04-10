import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import json

from object_detection.dataset.dataset_factory import dataset_factory
from object_detection.model.model_factory import model_factory
from object_detection.config.config_factory import config_factory
from tensorflow.contrib.eager.python import saver as eager_saver
from object_detection.utils.bbox_transform import decode_bbox_with_mean_and_std
from object_detection.utils.bbox_tf import bboxes_clip_filter

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
num_classes = 91


def eval_by_cocotools(res_file_path, mode, root_path):
    coco_gt = COCO(os.path.join(root_path, 'annotations', 'instances_{}2017.json'.format(mode)))
    coco_dt = coco_gt.loadRes(res_file_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    coco_eval.params.imgIds = coco_dt.getImgIds()
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def eval_coco(model,
              result_file_path,
              dataset_mode,
              image_format,
              preprocessing_type,
              root_path,
              config,
              min_size=10,
              ):
    """
    COCO Eval 的总体思路
    1. 构建tf.data.Dataset对象，返回需要测试COCO数据集的 preprocessed_image, raw_image_height, raw_image_width, image_id。
    2. 通过训练好的模型以及 preprocessed_image 获取预测结果，包括每张图片对应的 image_id, bboxes, classes，scores。
    2.1. 将预测结果保存为一个序列，序列中每个元素都是一个字典，分别包括image_id, category_id, bbox, score四个对象。
    2.2. image_id是int32，category_id是int32，bbox是个长度为4的float32数组，score是float32。
    2.3. 具体细节可以参考官方给出的实例：
    https://github.com/cocodataset/cocoapi/blob/master/results/instances_val2014_fakebbox100_results.json
    3. 通过COCOEval工具进行测试。
    3.1. 具体细节可以参考官方给出的实例：
    https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    3.2. 大概过程就是构建pycocotools.coco.COCO对象，导入结果数组、通过COCO.loadRes构建预测对象，最后通过cocoEval计算结果
    :param result_file_path:            path to save result json file
    :param model:                       pre-trained model
    :param dataset_mode:                train or val
    :param image_format:
    :param preprocessing_type:
    :param root_path:                   VOC的目录，要具体到某一年
    :param config:
    :param min_size:
    :return:
    """
    dataset_configs = {'root_dir': root_path,
                       'mode': dataset_mode,
                       'min_size': config['image_max_size'], 'max_size': config['image_min_size'],
                       'preprocessing_type': preprocessing_type,
                       'caffe_pixel_means': config['bgr_pixel_means']}
    dataset = dataset_factory(dataset_mode, mode=dataset_mode, **dataset_configs)

    res_list = []
    for img, img_scale, raw_h, raw_w, img_id in dataset:
        final_bboxes, final_category, final_scores = model(img, False)
        final_bboxes = final_bboxes / tf.to_float(img_scale)

        # scores, roi_txtytwth, rois = model.im_detect(img, img_scale)
        # roi_txtytwth = tf.reshape(roi_txtytwth, [-1, num_classes, 4])
        #
        # res_score = []
        # res_bbox = []
        # res_category = []
        # for j in range(1, num_classes):
        #     inds = tf.where(scores[:, j] > config['prediction_score_threshold'])[:, 0]
        #     cls_scores = tf.gather(scores[:, j], inds)
        #     cls_boxes = decode_bbox_with_mean_and_std(tf.gather(rois, inds),
        #                                               tf.gather(roi_txtytwth[:, j, :], inds),
        #                                               target_means=config['roi_proposal_means'],
        #                                               target_stds=config['roi_proposal_stds'])
        #
        #     cls_boxes, inds = bboxes_clip_filter(cls_boxes, 0, raw_h, raw_w, min_size)
        #     cls_scores = tf.gather(cls_scores, inds)
        #     keep = tf.image.non_max_suppression(cls_boxes, cls_scores, config['max_objects_per_class_per_image'],
        #                                         iou_threshold=config['prediction_score_threshold'])
        #     if tf.size(keep).numpy() == 0:
        #         continue
        #
        #     res_score.append(tf.gather(cls_scores, keep))
        #     res_bbox.append(tf.gather(cls_boxes, keep))
        #     res_category.append(tf.ones_like(keep, dtype=tf.int32) * j)
        #
        # scores_after_nms = tf.concat(res_score, axis=0)
        # bboxes_after_nms = tf.concat(res_bbox, axis=0)
        # category_after_nms = tf.concat(res_category, axis=0)
        #
        # final_scores, final_idx = tf.nn.top_k(scores_after_nms, k=tf.minimum(config['max_objects_per_image'],
        #                                                                      tf.size(scores_after_nms)),
        #                                       sorted=False)
        # final_bboxes = tf.gather(bboxes_after_nms, final_idx).numpy()
        # final_category = tf.gather(category_after_nms, final_idx).numpy()
        # final_scores = final_scores.numpy()

        for cur_bbox, cur_category, cur_score in zip(final_bboxes, final_category, final_scores):
            res_list.append({
                'image_id': int(img_id),
                'category_id': int(cur_category),
                'bbox': [float(cur_bbox[1]), float(cur_bbox[0]), float(cur_bbox[3]), float(cur_bbox[2])],
                'score': float(cur_score)
            })

    with open(result_file_path, 'w') as f:
        json.dump(res_list, f)
    eval_by_cocotools(result_file_path, dataset_mode, root_path)


def _load_from_ckpt_file(model, ckpt_file_path):
    saver = eager_saver.Saver(model.variables)
    for var in model.variables:
        tf.logging.info('restore var {}'.format(var.name))
    if tf.train.latest_checkpoint(ckpt_file_path) is not None:
        saver.restore(tf.train.latest_checkpoint(ckpt_file_path))
    else:
        raise ValueError('unknown ckpt file {}'.format(ckpt_file_path))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a Fast R-CNN model')
    parser.add_argument('ckpt_file_path', type=str, help='target ckpt file path', )

    parser.add_argument('--gpu_id', type=str, default='0')

    parser.add_argument('--dataset_mode', type=str, default='val', help='one of [train or val]')

    parser.add_argument('--model_type', type=str, default='faster_rcnn', help='one of [faster_rcnn, fpn]')
    parser.add_argument('--backbone', type=str, default='vgg16', help='one of [vgg16, resnet50, resnet101, resnet152]')

    parser.add_argument('--use_fpn_tensorflow_model', default=False, type=bool,
                        help='load fpn tensorflow model, only support resnet50 backbone')

    parser.add_argument('--root_path', help='path to pascal COCO',
                        default='/ssd/zhangyiyang/COCO2017', type=str)
    parser.add_argument('--result_file_dir', help='path to save detection result json file',
                        default='/ssd/zhangyiyang/results/', type=str)
    parser.add_argument('--logs_name', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def main(args):
    # 设置 eager 模式必须的参数
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)
    tf.logging.set_verbosity(tf.logging.INFO)

    # 获取模型并初始化参数
    model_config = config_factory('coco', args.model_type)
    cur_model = model_factory(args.model_type, args.backbone, model_config)
    preprocessing_type = 'caffe'
    cur_model(tf.to_float(np.random.rand(1, 800, 600, 3)), False)

    # result file path
    # {result_file_dir}/{model_type}/{backbone}/{logs_name}/coco_res.json
    logs_name = args.logs_name if args.logs_name is not None else 'default'
    final_result_file_dir = os.path.join(args.result_file_dir, args.model_type, args.backbone, logs_name)
    if not os.path.exists(final_result_file_dir):
        os.makedirs(final_result_file_dir)
    final_result_file_path = os.path.join(final_result_file_dir, 'coco_res.json')

    # 导入预训练模型
    image_format = 'bgr'
    if args.use_fpn_tensorflow_model:
        image_format = 'rgb'
        cur_model.load_fpn_tensorflow_weights(args.ckpt_file_path)
    else:
        _load_from_ckpt_file(cur_model, args.ckpt_file_path)

    # 将预测结果写到文件，并评估结果
    eval_coco(cur_model,
              result_file_path=final_result_file_path,
              dataset_mode=args.dataset_mode,
              image_format=image_format,
              preprocessing_type=preprocessing_type,
              root_path=os.path.join(args.root_path),
              config=model_config,)


if __name__ == '__main__':
    main(parse_args())
