from object_detection.dataset.coco_tf_dataset_generator import get_training_dataset as get_coco_train_dataset
from object_detection.dataset.coco_tf_dataset_generator import get_eval_dataset as get_coco_eval_dataset
from object_detection.dataset.pascal_tf_dataset_generator import get_dataset as get_pascal_train_dataset
from object_detection.dataset.eval_pascal_tf_dataset import get_dataset_by_local_file as get_pascal_eval_dataset


def dataset_factory(dataset_type, mode, configs):
    if dataset_type == 'pascal':
        if mode == 'train':
            return get_pascal_train_dataset(**configs)
        elif mode == 'test':
            return get_pascal_eval_dataset('test', **configs)
        raise ValueError('unknown mode {} for dataset type {}'.format(mode, dataset_type))

    if dataset_type == 'coco':
        if mode == 'train':
            return get_coco_train_dataset(**configs)
        elif mode == 'val':
            return get_coco_eval_dataset(**configs)
        raise ValueError('unknown mode {} for dataset type {}'.format(mode, dataset_type))

    raise ValueError('unknown dataset type {}'.format(dataset_type))
