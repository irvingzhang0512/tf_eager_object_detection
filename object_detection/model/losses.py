import tensorflow as tf

__all__ = ['get_rpn_loss', 'get_roi_loss']


def cls_loss(logits, labels, weight=1):
    """

    :param weight:
    :param logits: [num_anchors, 2]
    :param labels: [num_anchors, ]，取值[0, num_classes)(roi training) 或 [0, 1](rpn training)
    :return:
    """
    return tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=tf.to_int32(labels),
                                                  weights=weight)


def smooth_l1_loss(bbox_txtytwth_pred, bbox_txtytwth_gt, outside_weights=1, sigma=1.0):
    """

    :param bbox_txtytwth_pred:   [num_rpn_training_samples, 4]
    :param bbox_txtytwth_gt:     [num_rpn_training_samples, 4]
    :param outside_weights:
    :param sigma:
    :return:

    """
    sigma_2 = sigma ** 2
    box_diff = tf.to_float(bbox_txtytwth_pred) - tf.to_float(bbox_txtytwth_gt)
    abs_in_box_diff = tf.abs(box_diff)
    loss = tf.where(abs_in_box_diff < 1. / sigma_2,
                    tf.pow(box_diff, 2) * (sigma_2 / 2.),
                    (abs_in_box_diff - 0.5 / sigma_2)
                    )

    out_loss_box = outside_weights * loss
    loss_box = tf.reduce_sum(out_loss_box)
    return loss_box


def get_rpn_loss(rpn_raw_score, rpn_raw_pred_txtytwth, rpn_gt_labels, rpn_gt_txtytwth,
                 rpn_training_idx, rpn_pos_num,
                 sigma=3.0):
    """

    :param rpn_raw_score:               [num_anchors,]
    :param rpn_raw_pred_txtytwth:       [num_anchors, 4]
    :param rpn_gt_labels:               [num_rpn_training_samples, ]
    :param rpn_gt_txtytwth:             [num_rpn_pos_samples, 4]
    :param rpn_training_idx:            [num_rpn_training_samples, ]
    :param rpn_pos_num:                 scalar
    :param sigma:                       scalar
    :return:
    """
    rpn_cls_loss = cls_loss(tf.gather(rpn_raw_score, rpn_training_idx), rpn_gt_labels)
    if rpn_pos_num.numpy() == 0:
        rpn_reg_loss = tf.to_float(0)
    else:
        rpn_reg_loss = smooth_l1_loss(tf.gather(rpn_raw_pred_txtytwth, rpn_training_idx[:rpn_pos_num]),
                                      rpn_gt_txtytwth, sigma=sigma) / tf.to_float(tf.size(rpn_training_idx))
    return rpn_cls_loss, rpn_reg_loss


def get_roi_loss(roi_raw_score, roi_raw_pred_txtytwth, roi_gt_labels, roi_gt_txtytwth,
                 roi_training_idx, roi_pos_num,
                 sigma):
    """

    :param roi_raw_score:               [num_rois,]
    :param roi_raw_pred_txtytwth:       [num_rois, 4]
    :param roi_gt_labels:               [num_roi_training_samples, ]
    :param roi_gt_txtytwth:             [num_roi_pos_samples, 4]
    :param roi_training_idx:            [num_roi_training_samples, ]
    :param roi_pos_num:                 scalar
    :param sigma:                       scalar
    :return:
    """
    roi_training_idx = tf.to_int32(roi_training_idx)
    rpn_cls_loss = cls_loss(tf.gather(roi_raw_score, roi_training_idx), roi_gt_labels)

    pred_txtytwth = tf.gather_nd(roi_raw_pred_txtytwth,
                                 tf.stack([roi_training_idx[:roi_pos_num], roi_gt_labels[:roi_pos_num]], axis=1))

    if roi_pos_num.numpy() == 0:
        rpn_reg_loss = tf.to_float(0)
    else:
        rpn_reg_loss = smooth_l1_loss(pred_txtytwth,
                                      roi_gt_txtytwth,
                                      sigma=sigma) / tf.to_float(tf.size(roi_training_idx))
    return rpn_cls_loss, rpn_reg_loss
