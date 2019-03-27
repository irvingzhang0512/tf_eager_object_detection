import tensorflow as tf
import numpy as np
from tqdm import tqdm
from object_detection.dataset.eval_pascal_tf_dataset import get_dataset_by_local_file, get_dataset_by_tf_records
from object_detection.utils.bbox_transform import decode_bbox_with_mean_and_std
from object_detection.utils.bbox_tf import bboxes_clip_filter

num_classes = 21
class_list = ('__background__',  # always index 0
              'aeroplane', 'bicycle', 'bird', 'boat',  'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor')
class_name_to_id_dict = dict(list(zip(class_list, list(range(num_classes)))))
class_id_to_name_dict = dict(list(zip(list(range(num_classes)), class_list)))

__all__ = ['get_prediction_files']


def get_prediction_files(cur_model,
                         dataset_type='tf', image_format='bgr',
                         preprocessing_type='caffe', caffe_pixel_means=None,
                         min_edge=600, max_edge=1000,
                         data_root_path='/home/tensorflow05/data/VOCdevkit/VOC2007',
                         mode='test',
                         result_file_format='/home/tensorflow05/zyy/tf_eager_object_detection/results/{:s}.txt',
                         score_threshold=0.0, iou_threshold=0.5,
                         max_objects_per_class=50, max_objects_per_image=50,
                         target_means=None, target_stds=None,
                         min_size=10):
    """
    使用模型，生成预测结果文件
    :param cur_model:                   已导入pre-trained model的模型
    :param dataset_type:                预测数据集类型，有 cv 和 tf 两个选项
    :param image_format:
    :param caffe_pixel_means:
    :param preprocessing_type:
    :param min_edge:
    :param max_edge:
    :param data_root_path:              数据集所在位置
    :param mode:                        需要预测的数据集类型，train val trainval test
    :param result_file_format:          `result_file_format.format(class_name)` 就是对应类型输出结果文件具体路径
    :param score_threshold:             预测结果最小得分
    :param iou_threshold:               进行nms时的 iou threshold
    :param max_objects_per_class:       一张图中，每个类型最多能够输出多少个预测结果
    :param max_objects_per_image:       一张图片中，一共最多能生成多少预测结果
    :param target_means:                decode_bbox_with_mean_and_std 参数
    :param target_stds:                 decode_bbox_with_mean_and_std 参数
    :param min_size:                    最终结果最小边长（像素）
    :return:
    """
    if image_format not in ['bgr', 'rgb']:
        raise ValueError('unknown image format {}'.format(image_format))

    if dataset_type == 'cv2':
        eval_dataset, image_sets = get_dataset_by_local_file(mode, data_root_path,
                                                             image_format=image_format,
                                                             preprocessing_type=preprocessing_type,
                                                             caffe_pixel_means=caffe_pixel_means,
                                                             min_edge=min_edge, max_edge=max_edge)
    elif dataset_type == 'tf':
        eval_dataset, image_sets = get_dataset_by_tf_records(mode, data_root_path,
                                                             preprocessing_type=preprocessing_type,
                                                             caffe_pixel_means=caffe_pixel_means,
                                                             min_edge=min_edge, max_edge=max_edge)
    else:
        raise ValueError('unknown dataset type {}'.format(dataset_type))

    if target_stds is None:
        target_stds = [0.1, 0.1, 0.2, 0.2]
    if target_means is None:
        target_means = [0, 0, 0, 0]

    all_boxes = [[[] for _ in range(len(image_sets))]
                 for _ in range(num_classes)]
    i = 0
    for img, img_scale, raw_h, raw_w in tqdm(eval_dataset):
        raw_h = tf.to_float(raw_h)
        raw_w = tf.to_float(raw_w)
        scores, roi_txtytwth, rois = cur_model.im_detect(img, img_scale)
        roi_txtytwth = tf.reshape(roi_txtytwth, [-1, num_classes, 4])
        for j in range(1, num_classes):
            inds = tf.where(scores[:, j] > score_threshold)[:, 0]
            cls_scores = tf.gather(scores[:, j], inds)
            cls_boxes = decode_bbox_with_mean_and_std(tf.gather(rois, inds),
                                                      tf.gather(roi_txtytwth[:, j, :], inds),
                                                      target_means=target_means, target_stds=target_stds)
            cls_boxes, inds = bboxes_clip_filter(cls_boxes, 0, raw_h, raw_w, min_size)
            cls_scores = tf.gather(cls_scores, inds)
            keep = tf.image.non_max_suppression(cls_boxes, cls_scores, max_objects_per_class,
                                                iou_threshold=iou_threshold)

            cls_scores = cls_scores.numpy()
            cls_boxes = cls_boxes.numpy()
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            cls_dets = cls_dets[keep.numpy(), :]
            all_boxes[j][i] = cls_dets

        if max_objects_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, num_classes)])
            if len(image_scores) > max_objects_per_image:
                image_thresh = np.sort(image_scores)[-max_objects_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        i += 1

    for cls_ind, cls in enumerate(class_list):
        if cls == '__background__':
            continue
        tf.logging.info('Writing {} VOC results file'.format(cls))
        filename = result_file_format.format(cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(image_sets):
                dets = np.array(all_boxes[cls_ind][im_ind])
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))
