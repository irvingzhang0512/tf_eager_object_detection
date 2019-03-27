import os
import sys
import tensorflow as tf
import argparse
import object_detection.dataset.utils.tf_record_utils as dataset_utils
import object_detection.dataset.utils.label_map_utils as label_map_utils
from tqdm import tqdm
from lxml import etree


def _get_tf_example(xml_dict, label_map_dict, image_path):
    with open(image_path, 'rb') as image:
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
            xmin.append(float(int(obj['bndbox']['xmin']) - 1) / (width - 1))
            ymin.append(float(int(obj['bndbox']['ymin']) - 1) / (height - 1))
            xmax.append(float(int(obj['bndbox']['xmax']) - 1) / (width - 1))
            ymax.append(float(int(obj['bndbox']['ymax']) - 1) / (height - 1))
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
    writers = dataset_utils.get_multi_tf_record_writers(base_path=args.writer_base_path,
                                                        file_pattern=args.writer_file_pattern,
                                                        year=args.year,
                                                        number=args.writers_number,
                                                        mode=args.mode)
    label_map_dict = label_map_utils.get_label_map_dict(args.label_map_path)
    if args.year == "2007":
        years = ["VOC2007"]
    elif args.year == "2012":
        years = ["VOC2012"]
    elif args.year == "0712":
        years = ["VOC2007", "VOC2012"]
    else:
        raise ValueError('unknown year {}'.format(args.year))

    annotation_file_paths_list = []
    root_paths = []
    for year in years:
        with open(os.path.join(args.data_root_path, year, 'ImageSets', 'Main', 'aeroplane_%s.txt' % args.mode),
                  'r') as f:
            lines = f.readlines()
        cur_annotation_list = [
            os.path.join(args.data_root_path, year, 'Annotations', line.strip().split(' ')[0] + '.xml')
            for line in lines
        ]
        cur_root_paths = [os.path.join(args.data_root_path, year)] * len(lines)

        annotation_file_paths_list += cur_annotation_list
        root_paths += cur_root_paths

    for idx, (annotation_file_path, root_path) in enumerate(tqdm(zip(annotation_file_paths_list, root_paths))):
        with open(annotation_file_path, 'r') as f:
            xml_str = f.read()
        xml_dict = dataset_utils.recursive_parse_xml_to_dict(etree.fromstring(xml_str))['annotation']
        tf_example = _get_tf_example(xml_dict, label_map_dict,
                                     os.path.join(root_path, 'JPEGImages', xml_dict['filename']))
        writers[idx % args.writers_number].write(tf_example.SerializeToString())
    for writer in writers:
        writer.close()


def _parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="trainval")
    parser.add_argument('--year', type=str, default="2007", help="one of [2007, 2012, 0712]")
    parser.add_argument('--writer_file_pattern', type=str, default='pascal_%s_%s_%02d.tfrecords',
                        help='tf records output file name pattern')
    parser.add_argument('--writers_number', type=int, default=5, help='split tf records into several files.')

    parser.add_argument('--writer_base_path', type=str, default="/path/to/tf_eager_records",
                        help='path to save generated tf record files.')
    parser.add_argument('--label_map_path', type=str,
                        help='path to pascal_label_map.pbtxt, already exists in ./scripts/label_map_src/',
                        default='./scripts/label_map_src/pascal_label_map.pbtxt')
    parser.add_argument('--data_root_path', type=str, default='/path/to/VOCdevkit')

    # parser.add_argument('--writer_base_path', type=str, default="D:\\data\\VOCdevkit\\tf_eager_records")
    # parser.add_argument('--label_map_path', type=str,
    #                     help='path to pascal_label_map.pbtxt, already exists in ./scripts/label_map_src/',
    #                     default='./scripts/label_map_src/pascal_label_map.pbtxt')
    # parser.add_argument('--data_root_path', type=str, default='D:\\data\\VOCdevkit')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(_parse_arguments(sys.argv[1:]))
