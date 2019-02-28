import tensorflow as tf
import numpy as np
import cv2
import os
import object_detection.dataset.label_map_utils as label_map_utils
import object_detection.dataset.tf_record_utils as dataset_utils
from lxml import etree


def _read_image(file_path):
    img = cv2.imread(file_path).astype(np.float32)
    img -= np.array([[[102.9801, 115.9465, 122.7717]]])
    h, w, _ = img.shape
    min_edge = 600
    max_edge = 1000
    scale1 = min_edge / min(h, w)
    scale2 = max_edge / max(h, w)
    scale = min(scale1, scale2)
    img = cv2.resize(img, None, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_LINEAR)
    return img, scale


def get_dataset(mode, root_path, label_map_file_path):
    label_map_dict = label_map_utils.get_label_map_dict(label_map_file_path)
    with open(os.path.join(root_path, 'ImageSets', 'Main', 'aeroplane_%s.txt' % mode), 'r') as f:
        lines = f.readlines()
    examples_list = [line.strip().split(' ')[0] for line in lines]
    annotations_dir = os.path.join(root_path, 'Annotations')
    img_dir = os.path.join(root_path, 'JPEGImages')

    def _map_from_xml_and_cv2(example):
        example = example.decode()
        with open(os.path.join(annotations_dir, str(example) + '.xml'), 'r') as f:
            xml_str = f.read()
        xml_dict = dataset_utils.recursive_parse_xml_to_dict(etree.fromstring(xml_str))['annotation']
        img_file_path = os.path.join(img_dir, xml_dict['filename'])
        img, img_scale = _read_image(img_file_path)
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        if 'object' in xml_dict:
            for obj in xml_dict['object']:
                xmin.append((float(obj['bndbox']['xmin']) - 1) * img_scale)
                ymin.append((float(obj['bndbox']['ymin']) - 1) * img_scale)
                xmax.append((float(obj['bndbox']['xmax']) - 1) * img_scale)
                ymax.append((float(obj['bndbox']['ymax']) - 1) * img_scale)
                classes_text.append(obj['name'].encode('utf8'))
                classes.append(label_map_dict[obj['name']])

        return img, np.stack([ymin, xmin, ymax, xmax], axis=0).transpose().astype(np.float32), np.array(classes).astype(
            np.int32), np.array(classes_text)

    dataset = tf.data.Dataset.from_tensor_slices(examples_list).map(
        lambda example: tf.py_func(_map_from_xml_and_cv2,
                                   [example],
                                   [tf.float32, tf.float32, tf.int32, tf.string])
    ).batch(1)

    return dataset
