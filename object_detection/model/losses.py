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


def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    print(out_loss_box.shape)
    loss_box = tf.reduce_mean(tf.reduce_sum(
        out_loss_box,
        axis=dim
    ))
    return loss_box
