import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from object_detection.dataset.dataset_factory import dataset_factory
from object_detection.model.model_factory import model_factory
from object_detection.config.config_factory import config_factory
from tensorflow.contrib.eager.python import saver as eager_saver

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152


def eval_by_cocotools(res_list, mode):
    pass


def eval_coco(model,
              dataset_mode,
              image_format,
              preprocessing_type,
              root_path,
              config,
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
    :param model:                       导入好参数的模型
    :param dataset_mode:                train or test
    :param image_format:
    :param preprocessing_type:
    :param root_path:                   VOC的目录，要具体到某一年
    :param config:
    :return:
    """
    dataset_configs = {'root_dir': root_path,
                       'mode': dataset_mode,
                       'min_size': config['image_max_size'], 'max_size': config['image_min_size'],
                       'preprocessing_type': preprocessing_type,
                       'caffe_pixel_means': config['bgr_pixel_means']}
    dataset = dataset_factory(dataset_mode, mode=dataset_mode, **dataset_configs)
    # TODO: eval coco model
    pass


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

    parser.add_argument('--dataset_mode', type=str, default='test', help='one of [test, train, trainval, val]')

    parser.add_argument('--model_type', type=str, default='faster_rcnn', help='one of [faster_rcnn, fpn]')
    parser.add_argument('--backbone', type=str, default='vgg16', help='one of [vgg16, resnet50, resnet101, resnet152]')

    parser.add_argument('--use_fpn_tensorflow_model', default=False, type=bool,
                        help='load fpn tensorflow model, only support resnet50 backbone')

    parser.add_argument('--use_07_metric', default=True, type=bool)

    parser.add_argument('--root_path', help='path to pascal VOCdevkit',
                        default='/ssd/zhangyiyang/tf_eager_object_detection/VOCdevkit', type=str)

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

    # 导入预训练模型
    image_format = 'bgr'
    if args.use_fpn_tensorflow_model:
        image_format = 'rgb'
        cur_model.load_fpn_tensorflow_weights(args.ckpt_file_path)
    else:
        _load_from_ckpt_file(cur_model, args.ckpt_file_path)

    # 将预测结果写到文件，并评估结果
    eval_coco(cur_model,
              dataset_mode=args.dataset_mode,
              image_format=image_format,
              preprocessing_type=preprocessing_type,
              root_path=os.path.join(args.root_path),
              config=model_config)


if __name__ == '__main__':
    main(parse_args())
