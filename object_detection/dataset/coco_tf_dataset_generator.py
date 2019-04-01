import os
import numpy as np
import tensorflow as tf
from functools import partial
from pycocotools.coco import COCO

from object_detection.dataset.utils.tf_dataset_utils import image_argument_with_imgaug, preprocessing_func


_COCO_TRAIN_DATASET = None
_COCO_VAL_DATASET = None
_COCO_TEST_DATASET = None


class CocoDataset:
    def __init__(self, root_dir='D:\\data\\COCO2017', sub_dir='train',
                 min_edge=32,):
        if sub_dir not in ['train', 'val']:
            raise ValueError('unknown sub dir {}'.format(sub_dir))

        annotation_file_path = os.path.join(root_dir, 'annotations', 'instances_{}2017.json'.format(sub_dir))
        self._image_dir = os.path.join(root_dir, sub_dir+"2017")

        self._coco = COCO(annotation_file=annotation_file_path)
        self._get_cat_id_name_dict()
        self._img_ids, self._img_info_dict = self._filter_images(min_edge=min_edge)

    @property
    def img_ids(self):
        return self._img_ids

    @property
    def img_info_dict(self):
        return self._img_info_dict

    @property
    def cat_id_to_name_dict(self):
        return self._id_to_name_dict

    @property
    def cat_name_to_id_dict(self):
        return self._name_to_id_dict

    def _get_cat_id_name_dict(self):
        cat_ids = self._coco.getCatIds()
        id_to_name = {0: 'background'}
        name_to_id = {'background': 0}
        for cat_id in cat_ids:
            cat_name = self._coco.loadCats(cat_id)[0]['name']
            id_to_name[cat_id] = cat_name
            name_to_id[cat_name] = cat_id
        self._id_to_name_dict = id_to_name
        self._name_to_id_dict = name_to_id

    def _filter_images(self, min_edge):
        all_img_ids = list(set([_['image_id'] for _ in self._coco.anns.values()]))
        img_ids = []
        img_info_dict = {}
        for i in all_img_ids:
            info = self._coco.loadImgs(i)[0]

            ann_ids = self._coco.getAnnIds(imgIds=i)
            ann_info = self._coco.loadAnns(ann_ids)
            _, labels, _ = self._parse_ann_info(ann_info)

            if min(info['width'], info['height']) >= min_edge and labels.shape[0] != 0:
                img_ids.append(i)
                img_info_dict[i] = info
        return img_ids, img_info_dict

    def _parse_ann_info(self, ann_info):
        """Parse bbox annotation.

        Args
        ---
            ann_info (list[dict]): Annotation info of an image.

        Returns
        ---
            dict: A dict containing the following keys: bboxes,
                bboxes_ignore, labels.
        """
        gt_bboxes = []
        gt_labels = []
        gt_labels_text = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [y1, x1, y1 + h - 1, x1 + w - 1]
            gt_bboxes.append(bbox)
            gt_labels.append(ann['category_id'])
            gt_labels_text.append(self._id_to_name_dict[ann['category_id']])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_labels_text = np.array(gt_labels_text, dtype=np.string_)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_labels_text = np.array([], dtype=np.string_)

        return gt_bboxes, gt_labels, gt_labels_text

    def __getitem__(self, img_id):
        # 获取 annotation dict 信息
        ann_ids = self._coco.getAnnIds(imgIds=img_id)
        ann_info = self._coco.loadAnns(ann_ids)
        gt_bboxes, gt_labels, gt_labels_text = self._parse_ann_info(ann_info)

        # 设置 bboxes 范围为 [0, 1]
        image_height, image_width = self._img_info_dict[img_id]['height'], self._img_info_dict[img_id]['width']
        gt_bboxes[:, ::2] = gt_bboxes[:, ::2] / image_height
        gt_bboxes[:, 1::2] = gt_bboxes[:, 1::2] / image_width

        file_path = os.path.join(self._image_dir, self._img_info_dict[img_id]['file_name'])
        return file_path, gt_bboxes, image_height, image_width, gt_labels


def get_dataset(root_dir='D:\\data\\COCO2017',
                mode='train',
                min_size=600, max_size=1000,
                preprocessing_type='caffe',
                batch_size=1,
                repeat=1,
                shuffle=False, shuffle_buffer_size=1000,
                prefetch=False, prefetch_buffer_size=1000,
                argument=True, iaa_sequence=None):
    global _COCO_TRAIN_DATASET, _COCO_VAL_DATASET, _COCO_TEST_DATASET
    if mode not in ['train', 'val', 'test']:
        raise ValueError('unknown mode {}'.format(mode))
    if mode == 'train':
        if _COCO_TRAIN_DATASET is None:
            _COCO_TRAIN_DATASET = CocoDataset(root_dir=root_dir, sub_dir=mode)
        coco_dataset = _COCO_TRAIN_DATASET
    elif mode == 'val':
        if _COCO_VAL_DATASET is None:
            _COCO_VAL_DATASET = CocoDataset(root_dir=root_dir, sub_dir=mode)
        coco_dataset = _COCO_VAL_DATASET
    else:
        if _COCO_TEST_DATASET is None:
            _COCO_TEST_DATASET = CocoDataset(root_dir=root_dir, sub_dir=mode)
        coco_dataset = _COCO_TEST_DATASET

    def _parse_coco_data_py(img_id):
        file_path, gt_bboxes, image_height, image_width, gt_labels = coco_dataset[img_id]
        return file_path, gt_bboxes, image_height, image_width, gt_labels

    tf_dataset = tf.data.Dataset.from_tensor_slices(coco_dataset.img_ids).map(
        lambda img_id: tuple([*tf.py_func(_parse_coco_data_py, [img_id],
                                          [tf.string, tf.float32, tf.int64, tf.int64, tf.int64])])
    )
    tf_dataset = tf_dataset.map(
        lambda file_path, gt_bboxes, image_height, image_width, gt_labels, gt_labels_text: tuple([
            tf.image.decode_jpeg(tf.io.read_file(file_path), channels=3),
            gt_bboxes, image_height, image_width, gt_labels, gt_labels_text
        ])
    )

    if argument:
        image_argument_partial = partial(image_argument_with_imgaug, iaa_sequence=iaa_sequence)
        tf_dataset = tf_dataset.map(
            lambda image, bboxes, image_height, image_width, labels: tuple([
                *tf.py_func(image_argument_partial, [image, bboxes], [image.dtype, bboxes.dtype]),
                image_height, image_width, labels]),
            num_parallel_calls=5
        )

    preprocessing_partial_func = partial(preprocessing_func,
                                         min_size=min_size, max_size=max_size,
                                         preprocessing_type=preprocessing_type)

    tf_dataset = tf_dataset.batch(batch_size=batch_size).map(preprocessing_partial_func, num_parallel_calls=5)

    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=shuffle_buffer_size)
    if prefetch:
        tf_dataset = tf_dataset.prefetch(buffer_size=prefetch_buffer_size)

    return tf_dataset.repeat(repeat)
