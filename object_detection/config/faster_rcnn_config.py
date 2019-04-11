def get_default_pascal_faster_rcnn_config():
    return {
        # vgg16
        'vgg16_roi_feature_size': (7, 7, 512),
        'roi_head_keep_dropout_rate': 0.5,
        'vgg16_roi_pooling_max_pooling_flag': True,

        # resnet
        'resnet_roi_feature_size': (7, 7, 1024),
        'resnet_roi_pooling_max_pooling_flag': False,

        # base configs
        'num_classes': 21,
        'weight_decay': 0.0001,

        # anchors configs
        'ratios': [0.5, 1.0, 2.0],
        'scales': [8, 16, 32],
        'extractor_stride': 16,

        # training configs
        'learning_rate_multi_decay_steps': [80000],  # 50000 for pascal 2007, 80000 for pascal 0712
        'learning_rate_multi_lrs': [1e-3, 1e-4],
        'learning_rate_bias_double': True,
        'optimizer_momentum': 0.9,
        'epochs': 8,  # 14 for pascal 2007, 8 for pascal 0712

        # preprocessing configs
        'image_max_size': 1000,
        'image_min_size': 600,
        'bgr_pixel_means': [103.939, 116.779, 123.68],
        # 'bgr_pixel_means': [102.9801, 115.9465, 122.7717],  # for tf-faster-rcnn

        # predict & evaluate configs
        'evaluate_iou_threshold': 0.5,  # 计算map时使用，pred与gt的iou大于该阈值，则当前pred为TP，否则为FP
        'max_objects_per_class_per_image': 50,
        'max_objects_per_image': 50,
        'predictions_nms_iou_threshold': 0.3,
        'prediction_score_threshold': 0.0,
        'show_image_score_threshold': 0.3,  # 用于图像展示

        # anchor target & region proposal
        'rpn_proposal_means': [0, 0, 0, 0],
        'rpn_proposal_stds': [1.0, 1.0, 1.0, 1.0],

        # anchor target
        'rpn_sigma': 3.0,
        'rpn_pos_iou_threshold': 0.7,
        'rpn_neg_iou_threshold': 0.3,
        'rpn_total_sample_number': 256,
        'rpn_pos_sample_max_number': 128,

        # region proposal
        'rpn_proposal_train_pre_nms_sample_number': 12000,
        'rpn_proposal_train_after_nms_sample_number': 2000,
        'rpn_proposal_test_pre_nms_sample_number': 6000,
        'rpn_proposal_test_after_nms_sample_number': 300,
        'rpn_proposal_nms_iou_threshold': 0.7,

        # proposal target & prediction
        'roi_proposal_means': [0, 0, 0, 0],
        'roi_proposal_stds': [0.1, 0.1, 0.2, 0.2],

        # roi pooling
        'roi_pooling_size': 7,

        # proposal target
        'roi_sigma': 1.0,
        'roi_pos_iou_threshold': 0.5,
        'roi_neg_iou_threshold': 0.,
        'roi_total_sample_number': 128,
        'roi_pos_sample_max_number': 32,

    }


def get_default_coco_faster_rcnn_config():
    return {
        # vgg16
        'vgg16_roi_feature_size': (7, 7, 512),
        'roi_head_keep_dropout_rate': 0.5,
        'vgg16_roi_pooling_max_pooling_flag': True,

        # resnet
        'resnet_roi_feature_size': (7, 7, 1024),
        'resnet_roi_pooling_max_pooling_flag': False,

        # base configs
        'num_classes': 81,
        'weight_decay': 0.0001,

        # anchors configs
        'ratios': [0.5, 1.0, 2.0],
        'scales': [4, 8, 16, 32],
        'extractor_stride': 16,

        # training configs
        'learning_rate_multi_decay_steps': [470000],
        'learning_rate_multi_lrs': [1e-3, 1e-4],
        'learning_rate_bias_double': True,
        'optimizer_momentum': 0.9,
        'epochs': 6,

        # preprocessing configs
        'image_max_size': 1000,
        'image_min_size': 600,
        'bgr_pixel_means': [103.939, 116.779, 123.68],

        # predict & evaluate configs
        'evaluate_iou_threshold': 0.5,  # 计算map时使用，pred与gt的iou大于该阈值，则当前pred为TP，否则为FP
        'max_objects_per_class_per_image': 50,
        'max_objects_per_image': 50,
        'predictions_nms_iou_threshold': 0.3,
        'prediction_score_threshold': 0.0,
        'show_image_score_threshold': 0.3,  # 用于图像展示

        # anchor target & region proposal
        'rpn_proposal_means': [0, 0, 0, 0],
        'rpn_proposal_stds': [1.0, 1.0, 1.0, 1.0],

        # anchor target
        'rpn_sigma': 3.0,
        'rpn_pos_iou_threshold': 0.7,
        'rpn_neg_iou_threshold': 0.3,
        'rpn_total_sample_number': 256,
        'rpn_pos_sample_max_number': 128,

        # region proposal
        'rpn_proposal_train_pre_nms_sample_number': 12000,
        'rpn_proposal_train_after_nms_sample_number': 2000,
        'rpn_proposal_test_pre_nms_sample_number': 6000,
        'rpn_proposal_test_after_nms_sample_number': 300,
        'rpn_proposal_nms_iou_threshold': 0.7,

        # proposal target & prediction
        'roi_proposal_means': [0, 0, 0, 0],
        'roi_proposal_stds': [0.1, 0.1, 0.2, 0.2],

        # roi pooling
        'roi_pooling_size': 7,

        # proposal target
        'roi_sigma': 1.0,
        'roi_pos_iou_threshold': 0.5,
        'roi_neg_iou_threshold': 0.,
        'roi_total_sample_number': 128,
        'roi_pos_sample_max_number': 32,
    }


PASCAL_CONFIG = get_default_pascal_faster_rcnn_config()
COCO_CONFIG = get_default_coco_faster_rcnn_config()
