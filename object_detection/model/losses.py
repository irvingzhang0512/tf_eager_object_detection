import tensorflow as tf


def cls_loss(logits, labels, weight=1):
    """

    :param weight:
    :param logits: [num_anchors, 2]
    :param labels: [num_anchors, ]，取值[0, num_classes)
    :return:
    """
    return tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=tf.to_int32(labels),
                                                  weights=weight)


def smooth_l1_loss(bbox_txtytwth_pred, bbox_txtytwth_gt, inside_weights, outside_weights=1, sigma=1.0, dim=None):
    if dim is None:
        dim = [1]
    sigma_2 = sigma ** 2
    box_diff = bbox_txtytwth_pred - bbox_txtytwth_gt
    in_box_diff = inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    loss_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * loss_sign + \
                  (abs_in_box_diff - (0.5 / sigma_2)) * (1. - loss_sign)
    out_loss_box = outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box, axis=dim))
    return loss_box
