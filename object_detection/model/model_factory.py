from object_detection.model.fpn.resnet_fpn import ResnetV1Fpn
from object_detection.model.faster_rcnn.resnet_faster_rcnn import ResNetFasterRcnn
from object_detection.model.faster_rcnn.vgg16_faster_rcnn import Vgg16FasterRcnn

__all__ = ['model_factory']


def model_factory(model_type, backbone, config):
    if model_type == 'faster_rcnn':
        if backbone == 'vgg16':
            return _get_faster_rcnn_vgg16_model(None, config)
        elif backbone == 'resnet50':
            return _get_faster_rcnn_resnet_model(50, config)
        elif backbone == 'resnet101':
            return _get_faster_rcnn_resnet_model(101, config)
        elif backbone == 'resnet152':
            return _get_faster_rcnn_resnet_model(152, config)
        else:
            raise ValueError('unknown backbone {}'.format(backbone))
    elif model_type == 'fpn':
        if backbone == 'resnet50':
            return _get_fpn_resnet_model(50, config)
        elif backbone == 'resnet101':
            return _get_fpn_resnet_model(101, config)
        elif backbone == 'resnet152':
            return _get_fpn_resnet_model(152, config)
        else:
            raise ValueError('unknown backbone {}'.format(backbone))
    else:
        raise ValueError('unknown model type {}'.format(model_type))


def _get_fpn_resnet_model(depth, config):
    return ResnetV1Fpn(
        depth=depth,
        roi_head_keep_dropout_rate=config['roi_head_keep_dropout_rate'],

        roi_feature_size=config['resnet_roi_feature_size'],
        num_classes=config['num_classes'],
        weight_decay=config['weight_decay'],

        level_name_list=config['level_name_list'],
        min_level=config['min_level'],
        max_level=config['max_level'],
        top_down_dims=config['top_down_dims'],

        anchor_stride_list=config['anchor_stride_list'],
        base_anchor_size_list=config['base_anchor_size_list'],
        ratios=config['ratios'],
        scales=config['scales'],

        rpn_proposal_means=config['rpn_proposal_means'],
        rpn_proposal_stds=config['rpn_proposal_stds'],

        rpn_proposal_num_pre_nms_train=config['rpn_proposal_train_pre_nms_sample_number'],
        rpn_proposal_num_post_nms_train=config['rpn_proposal_train_after_nms_sample_number'],
        rpn_proposal_num_pre_nms_test=config['rpn_proposal_test_pre_nms_sample_number'],
        rpn_proposal_num_post_nms_test=config['rpn_proposal_test_after_nms_sample_number'],
        rpn_proposal_nms_iou_threshold=config['rpn_proposal_nms_iou_threshold'],

        rpn_sigma=config['rpn_sigma'],
        rpn_training_pos_iou_threshold=config['rpn_pos_iou_threshold'],
        rpn_training_neg_iou_threshold=config['rpn_neg_iou_threshold'],
        rpn_training_total_num_samples=config['rpn_total_sample_number'],
        rpn_training_max_pos_samples=config['rpn_pos_sample_max_number'],

        roi_proposal_means=config['roi_proposal_means'],
        roi_proposal_stds=config['roi_proposal_stds'],

        roi_pool_size=config['roi_pooling_size'],
        roi_pooling_max_pooling_flag=config['roi_pooling_max_pooling_flag'],

        roi_sigma=config['roi_sigma'],
        roi_training_pos_iou_threshold=config['roi_pos_iou_threshold'],
        roi_training_neg_iou_threshold=config['roi_neg_iou_threshold'],
        roi_training_total_num_samples=config['roi_total_sample_number'],
        roi_training_max_pos_samples=config['roi_pos_sample_max_number'],

        prediction_max_objects_per_image=config['max_objects_per_image'],
        prediction_max_objects_per_class=config['max_objects_per_class_per_image'],
        prediction_nms_iou_threshold=config['predictions_nms_iou_threshold'],
        prediction_score_threshold=config['prediction_score_threshold'],
    )


def _get_faster_rcnn_resnet_model(depth, config):
    return ResNetFasterRcnn(
        depth=depth,
        roi_feature_size=config['resnet_roi_feature_size'],

        num_classes=config['num_classes'],
        weight_decay=config['weight_decay'],

        ratios=config['ratios'],
        scales=config['scales'],
        extractor_stride=config['extractor_stride'],

        rpn_proposal_means=config['rpn_proposal_means'],
        rpn_proposal_stds=config['rpn_proposal_stds'],

        rpn_proposal_num_pre_nms_train=config['rpn_proposal_train_pre_nms_sample_number'],
        rpn_proposal_num_post_nms_train=config['rpn_proposal_train_after_nms_sample_number'],
        rpn_proposal_num_pre_nms_test=config['rpn_proposal_test_pre_nms_sample_number'],
        rpn_proposal_num_post_nms_test=config['rpn_proposal_test_after_nms_sample_number'],
        rpn_proposal_nms_iou_threshold=config['rpn_proposal_nms_iou_threshold'],

        rpn_sigma=config['rpn_sigma'],
        rpn_training_pos_iou_threshold=config['rpn_pos_iou_threshold'],
        rpn_training_neg_iou_threshold=config['rpn_neg_iou_threshold'],
        rpn_training_total_num_samples=config['rpn_total_sample_number'],
        rpn_training_max_pos_samples=config['rpn_pos_sample_max_number'],

        roi_proposal_means=config['roi_proposal_means'],
        roi_proposal_stds=config['roi_proposal_stds'],

        roi_pool_size=config['roi_pooling_size'],
        roi_pooling_max_pooling_flag=config['resnet_roi_pooling_max_pooling_flag'],

        roi_sigma=config['roi_sigma'],
        roi_training_pos_iou_threshold=config['roi_pos_iou_threshold'],
        roi_training_neg_iou_threshold=config['roi_neg_iou_threshold'],
        roi_training_total_num_samples=config['roi_total_sample_number'],
        roi_training_max_pos_samples=config['roi_pos_sample_max_number'],

        prediction_max_objects_per_image=config['max_objects_per_image'],
        prediction_max_objects_per_class=config['max_objects_per_class_per_image'],
        prediction_nms_iou_threshold=config['predictions_nms_iou_threshold'],
        prediction_score_threshold=config['prediction_score_threshold'],
    )


def _get_faster_rcnn_vgg16_model(slim_ckpt_file_path, config):
    return Vgg16FasterRcnn(
        slim_ckpt_file_path=slim_ckpt_file_path,
        roi_head_keep_dropout_rate=config['roi_head_keep_dropout_rate'],
        roi_feature_size=config['vgg16_roi_feature_size'],

        num_classes=config['num_classes'],
        weight_decay=config['weight_decay'],

        ratios=config['ratios'],
        scales=config['scales'],
        extractor_stride=config['extractor_stride'],

        rpn_proposal_means=config['rpn_proposal_means'],
        rpn_proposal_stds=config['rpn_proposal_stds'],

        rpn_proposal_num_pre_nms_train=config['rpn_proposal_train_pre_nms_sample_number'],
        rpn_proposal_num_post_nms_train=config['rpn_proposal_train_after_nms_sample_number'],
        rpn_proposal_num_pre_nms_test=config['rpn_proposal_test_pre_nms_sample_number'],
        rpn_proposal_num_post_nms_test=config['rpn_proposal_test_after_nms_sample_number'],
        rpn_proposal_nms_iou_threshold=config['rpn_proposal_nms_iou_threshold'],

        rpn_sigma=config['rpn_sigma'],
        rpn_training_pos_iou_threshold=config['rpn_pos_iou_threshold'],
        rpn_training_neg_iou_threshold=config['rpn_neg_iou_threshold'],
        rpn_training_total_num_samples=config['rpn_total_sample_number'],
        rpn_training_max_pos_samples=config['rpn_pos_sample_max_number'],

        roi_proposal_means=config['roi_proposal_means'],
        roi_proposal_stds=config['roi_proposal_stds'],

        roi_pool_size=config['roi_pooling_size'],
        roi_pooling_max_pooling_flag=config['vgg16_roi_pooling_max_pooling_flag'],

        roi_sigma=config['roi_sigma'],
        roi_training_pos_iou_threshold=config['roi_pos_iou_threshold'],
        roi_training_neg_iou_threshold=config['roi_neg_iou_threshold'],
        roi_training_total_num_samples=config['roi_total_sample_number'],
        roi_training_max_pos_samples=config['roi_pos_sample_max_number'],

        prediction_max_objects_per_image=config['max_objects_per_image'],
        prediction_max_objects_per_class=config['max_objects_per_class_per_image'],
        prediction_nms_iou_threshold=config['predictions_nms_iou_threshold'],
        prediction_score_threshold=config['prediction_score_threshold'],
    )
