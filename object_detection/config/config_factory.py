
def config_factory(data_type, model_type):
    if model_type == 'faster_rcnn':
        if data_type == 'pascal':
            from object_detection.config.faster_rcnn_config import PASCAL_CONFIG
            return PASCAL_CONFIG
        elif data_type == 'coco':
            from object_detection.config.faster_rcnn_config import COCO_CONFIG
            return COCO_CONFIG
    elif model_type == 'fpn':
        if data_type == 'pascal':
            from object_detection.config.fpn_config import PASCAL_CONFIG
            return PASCAL_CONFIG

    raise ValueError('config for dataset type {} and model type {} doesn\'t exist'.format(data_type, model_type))