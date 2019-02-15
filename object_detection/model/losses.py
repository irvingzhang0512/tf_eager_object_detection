import tensorflow as tf


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
