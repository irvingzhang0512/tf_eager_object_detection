import os
import sys
import tensorflow as tf
import argparse
import object_detection.dataset.tf_record_utils as dataset_utils
import object_detection.dataset.label_map_utils as label_map_utils
from tqdm import tqdm
from lxml import etree


def _get_tf_example(xml_dict, label_map_dict, image_root):
    with open(os.path.join(image_root, xml_dict['filename']), 'rb') as image:
        encoded_jpg = image.read()
        width = int(xml_dict['size']['width'])
        height = int(xml_dict['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    if 'object' in xml_dict:
        for obj in xml_dict['object']:
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_utils.int64_feature(height),
        'image/width': dataset_utils.int64_feature(width),
        'image/filename': dataset_utils.bytes_feature(xml_dict['filename'].encode('utf8')),
        'image/encoded': dataset_utils.bytes_feature(encoded_jpg),
        'image/object/bbox/xmin': dataset_utils.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_utils.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_utils.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_utils.float_list_feature(ymax),
        'image/object/class/label': dataset_utils.int64_list_feature(classes),
        'image/object/class/text': dataset_utils.bytes_list_feature(classes_text),
    }))
    return example


def main(args):
    writers = dataset_utils.get_multi_tf_record_writers(args.writer_base_path,
                                                        args.writer_file_patther,
                                                        args.writers_number,
                                                        args.mode)
    label_map_dict = label_map_utils.get_label_map_dict(args.label_map_path)
    with open(os.path.join(args.data_root_path, 'ImageSets', 'Main', 'aeroplane_%s.txt' % args.mode), 'r') as f:
        lines = f.readlines()
    examples_list = [line.strip().split(' ')[0] for line in lines]
    annotations_dir = os.path.join(args.data_root_path, 'Annotations')
    for idx, example in enumerate(tqdm(examples_list)):
        with open(os.path.join(annotations_dir, example + '.xml'), 'r') as f:
            xml_str = f.read()
        xml_dict = dataset_utils.recursive_parse_xml_to_dict(etree.fromstring(xml_str))['annotation']
        tf_example = _get_tf_example(xml_dict, label_map_dict, os.path.join(args.data_root_path, 'JPEGImages'))
        writers[idx % args.writers_number].write(tf_example.SerializeToString())
    for writer in writers:
        writer.close()


def _parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--writer_base_path', type=str, default="/home/tensorflow05/data/VOCdevkit/tf_eager_records")
    parser.add_argument('--writer_file_patther', type=str, default='pascal_%s_%02d.tfrecords')
    parser.add_argument('--writers_number', type=int, default=1)
    parser.add_argument('--label_map_path', type=str, default='/home/tensorflow05/zyy/tf_eager_object_detection/object_detection/dataset/label_map_src/pascal_label_map.pbtxt')
    parser.add_argument('--data_root_path', type=str, default='/home/tensorflow05/data/VOCdevkit/VOC2012')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(_parse_arguments(sys.argv[1:]))
