def get_default_faster_rcnn_config():
    return {
        'num_classes': 21,
        'extractor_stride': 16,
        'epochs': 14,

        'max_size': 2000,
        'min_size': 600,
        'ratios': [0.5, 1.0, 2.0],
        'scales': [8, 16, 32],

        'learning_rate_start': 1e-3,
        'learning_rate_decay_epochs': 9,
        'weight_decay': 0.0005,

        'rpn_cls_loss_weight': 1.0,
        'rpn_reg_loss_weight': 3.0,
        'rpn_pos_iou_threshold': 0.7,
        'rpn_neg_iou_threshold': 0.3,
        'rpn_total_sample_number': 256,
        'rpn_pos_sample_max_number': 128,
        'rpn_proposal_train_pre_nms_sample_number': 12000,
        'rpn_proposal_train_after_nms_sample_number': 2000,
        'rpn_proposal_test_pre_nms_sample_number': 6000,
        'rpn_proposal_test_after_nms_sample_number': 300,
        'rpn_proposal_nms_iou_threshold': 0.7,

        'roi_pooling_size': 7,
        'roi_cls_loss_weight': 1.0,
        'roi_reg_loss_weight': 1.0,
        'roi_pos_iou_threshold': 0.5,
        'roi_neg_iou_threshold': 0.1,
        'roi_total_sample_number': 128,
        'roi_pos_sample_max_number': 32,

    }


CONFIG = get_default_faster_rcnn_config()
