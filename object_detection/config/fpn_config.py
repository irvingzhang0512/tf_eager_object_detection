def get_default_pascal_faster_rcnn_config():
    return {
        # 不同backbone参数
        'resnet_roi_feature_size': [7, 7, 256],
        'roi_head_keep_dropout_rate': 0.5,

        # base configs
        'num_classes': 21,

        # fpn 特有参数
        'level_name_list': ['p2', 'p3', 'p4', 'p5', 'p6'],
        'min_level': 2,
        'max_level': 5,
        'top_down_dims': 256,

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

        # anchors configs
        'ratios': [0.5, 1.0, 2.0],
        'scales': [1.],
        'anchor_stride_list': [4, 8, 16, 32, 64],
        'base_anchor_size_list': [32, 64, 128, 256, 512],

        # training configs
        'learning_rate_multi_decay_steps': [60000, 80000],
        'learning_rate_multi_lrs': [1e-3, 1e-4, 1e-5],
        'optimizer_momentum': 0.9,
        'learning_rate_bias_double': False,
        'weight_decay': 0.0001,
        'epochs': 30,

        # rpn net configs
        'rpn_proposal_means': [0, 0, 0, 0],
        'rpn_proposal_stds': [1.0, 1.0, 1.0, 1.0],
        'rpn_sigma': 3.0,
        'rpn_pos_iou_threshold': 0.7,
        'rpn_neg_iou_threshold': 0.3,
        'rpn_total_sample_number': 256,
        'rpn_pos_sample_max_number': 128,
        'rpn_proposal_train_pre_nms_sample_number': 12000,
        'rpn_proposal_train_after_nms_sample_number': 2000,
        'rpn_proposal_test_pre_nms_sample_number': 6000,
        'rpn_proposal_test_after_nms_sample_number': 1000,
        'rpn_proposal_nms_iou_threshold': 0.7,

        'roi_pooling_size': 7,
        'roi_pooling_max_pooling_flag': True,

        # roi net configs
        'roi_proposal_means': [0, 0, 0, 0],
        'roi_proposal_stds': [0.1, 0.1, 0.2, 0.2],
        'roi_sigma': 1.0,
        'roi_pos_iou_threshold': 0.5,
        'roi_neg_iou_threshold': 0.,
        'roi_total_sample_number': 256,
        'roi_pos_sample_max_number': 64,

    }


PASCAL_CONFIG = get_default_pascal_faster_rcnn_config()
