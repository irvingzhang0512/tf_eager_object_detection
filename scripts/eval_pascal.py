import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from object_detection.model.model_factory import model_factory
from object_detection.config.config_factory import config_factory
from object_detection.evaluation.pascal_eval_files_utils import get_prediction_files
from object_detection.evaluation.detectron_pascal_evaluation_utils import voc_eval
from tensorflow.contrib.eager.python import saver as eager_saver

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
num_classes = 21,
class_list = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor')


def eval_from_scratch(model,
                      dataset_type,
                      dataset_mode,
                      image_format,
                      preprocessing_type,
                      root_path,
                      result_file_format,
                      cache_dir,
                      use_07_metric,
                      config,
                      ):
    """

    :param model:                       导入好参数的模型
    :param dataset_type:                训练时使用的原始数据是通过 cv2 产生还是 tf 产生
    :param dataset_mode:                train, val, test or trainval
    :param image_format:
    :param preprocessing_type:
    :param root_path:                   VOC的目录，要具体到某一年
    :param result_file_format:          要将test结果写到文件中，文件路径为 result_file_format.format(class_name)
    :param cache_dir:                   预测时，会将gt的信息使用pickle进行保存，保存的路径就是 cache_dir+'test_annots.pkl'
    :param use_07_metric:
    :param config:
    :return:
    """

    # 生成检测结果的本地文件
    get_prediction_files(model,
                         dataset_type=dataset_type,
                         image_format=image_format,
                         preprocessing_type=preprocessing_type,
                         caffe_pixel_means=config['bgr_pixel_means'],
                         min_edge=config['image_min_size'],
                         max_edge=config['image_max_size'],
                         data_root_path=root_path,
                         mode=dataset_mode,
                         result_file_format=result_file_format,
                         score_threshold=config['prediction_score_threshold'],
                         iou_threshold=config['predictions_nms_iou_threshold'],
                         max_objects_per_class=config['max_objects_per_class_per_image'],
                         max_objects_per_image=config['max_objects_per_image'] ,
                         target_means=config['roi_proposal_means'],
                         target_stds=config['roi_proposal_stds'],
                         min_size=10
                         )

    # 通过本地文件（包括检测结果和真实结果）计算map
    eval_by_local_files_and_gt_xmls(root_path,
                                    result_file_format,
                                    cache_dir,
                                    dataset_mode,
                                    config['evaluate_iou_threshold'],
                                    use_07_metric=use_07_metric, )


def eval_by_local_files_and_gt_xmls(root_path,
                                    result_file_format,
                                    cache_dir,
                                    mode,
                                    prediction_iou_threshold,
                                    use_07_metric=True):
    annotation_file_format = os.path.join(root_path, 'Annotations', "{}.xml")
    imagesetfile = os.path.join(root_path, 'ImageSets', 'Main', '{}.txt'.format(mode))
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
                           use_07_metric=use_07_metric,
                           )
        tf.logging.info('class {} get ap {}'.format(cls_name, cur_res[2]))
        all_ap += cur_res[2]
    tf.logging.info('map {}'.format(all_ap / (len(class_list) - 1)))


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

    parser.add_argument('--dataset_type', help='type of dataset, cv2 or tf', default='cv2', type=str)
    parser.add_argument('--dataset_mode', type=str, default='test', help='one of [test, train, trainval, val]')
    parser.add_argument('--year', type=str, default='2007', help='one of [2007, 2012]')

    parser.add_argument('--model_type', type=str, default='faster_rcnn', help='one of [faster_rcnn, fpn]')
    parser.add_argument('--backbone', type=str, default='vgg16', help='one of [vgg16, resnet50, resnet101, resnet152]')

    parser.add_argument('--use_tf_faster_rcnn_model', type=bool, default=False,
                        help='load tf-faster-rcnn model, only support resnet101 backbone')
    parser.add_argument('--use_fpn_tensorflow_model', default=False, type=bool,
                        help='load fpn tensorflow model, only support resnet50 backbone')
    parser.add_argument('--use_local_result_files', default=False, type=bool)

    parser.add_argument('--use_07_metric', default=True, type=bool)

    # parser.add_argument('--root_path', help='path to pascal VOCdevkit',
    #                     default='D:\\data\\VOCdevkit', type=str)
    # parser.add_argument('--result_file_format', help='local detection result file pattern',
    #                     default='D:\\data\\VOCdevkit\\VOC2007\\results\\{:s}.txt', type=str)
    # parser.add_argument('--annotation_cache_dir', help='path to save annotation cache pickle file',
    #                     default='D:\\data\\VOCdevkit\\VOC2007\\results', type=str)

    parser.add_argument('--root_path', help='path to pascal VOCdevkit',
                        default='/ssd/zhangyiyang/tf_eager_object_detection/VOCdevkit', type=str)
    parser.add_argument('--annotation_cache_dir', help='path to save annotation cache pickle file',
                        default='/ssd/zhangyiyang/tf_eager_object_detection/results', type=str)

    # path to save detection result files
    parser.add_argument('--result_file_dir', help='local detection result file pattern',
                        default='/ssd/zhangyiyang/tf_eager_object_detection/results', type=str)
    parser.add_argument('--logs_name', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def main(args):
    model_config = config_factory('pascal', args.model_type)

    # get result file format
    # {args.result_file_dir}/{args.model_type}/{args.backbone}/{logs_name}/{}.txt
    logs_name = args.logs_name if args.logs_name is not None else 'default'
    result_file_dir = os.path.join(args.result_file_dir, args.model_type, args.backbone, logs_name)
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir)
    result_file_path = os.path.join(result_file_dir, '{}.txt')

    if args.use_local_result_files:
        # 本地文件已存在，通过本地文件进行评估
        eval_by_local_files_and_gt_xmls(root_path=args.root_path,
                                        result_file_format=result_file_path,
                                        cache_dir=args.annotation_cache_dir,
                                        mode=args.dataset_mode,
                                        prediction_iou_threshold=model_config['evaluate_iou_threshold']
                                        )
        return

    # 判断参数合法性
    if args.year not in ['2007', '2012']:
        raise ValueError('unknown pascal year {}'.format(args.year))

    # 设置 eager 模式必须的参数
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)
    tf.logging.set_verbosity(tf.logging.INFO)

    # 获取模型并初始化参数
    cur_model = model_factory(args.model_type, args.backbone, model_config)
    preprocessing_type = 'caffe'
    cur_model(tf.to_float(np.random.rand(1, 800, 600, 3)), False)

    # 导入预训练模型
    image_format = 'bgr'
    if args.use_tf_faster_rcnn_model:
        cur_model.load_tf_faster_rcnn_tf_weights(args.ckpt_file_path)
    elif args.use_fpn_tensorflow_model:
        image_format = 'rgb'
        cur_model.load_fpn_tensorflow_weights(args.ckpt_file_path)
    else:
        _load_from_ckpt_file(cur_model, args.ckpt_file_path)

    # 将预测结果写到文件，并评估结果
    eval_from_scratch(cur_model,
                      dataset_type=args.dataset_type,
                      dataset_mode=args.dataset_mode,
                      image_format=image_format,
                      preprocessing_type=preprocessing_type,
                      root_path=os.path.join(args.root_path, 'VOC' + str(args.year)),
                      result_file_format=result_file_path,
                      cache_dir=args.annotation_cache_dir,
                      use_07_metric=args.use_07_metric,
                      config=model_config)


if __name__ == '__main__':
    main(parse_args())
