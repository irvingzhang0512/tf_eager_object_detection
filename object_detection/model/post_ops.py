import tensorflow as tf

from object_detection.utils.bbox_transform import decode_bbox_with_mean_and_std
from object_detection.utils.bbox_tf import bboxes_clip_filter as bboxes_clip_filter_tf


def predict_after_roi(roi_scores_softmax, roi_txtytwth, rois, image_shape,
                      target_means, target_stds,
                      max_num_per_class=5,
                      max_num_per_image=5,
                      nms_iou_threshold=0.3,
                      score_threshold=0.3,
                      extractor_stride=16,
                      ):
    """
    copy from https://github.com/Viredery/tf-eager-fasterrcnn/blob/master/detection/models/bbox_heads/bbox_head.py
    :param roi_scores_softmax:
    :param roi_txtytwth:
    :param rois:
    :param image_shape:
    :param target_means:
    :param target_stds:
    :param max_num_per_class:
    :param max_num_per_image:
    :param nms_iou_threshold:
    :param score_threshold:
    :param extractor_stride:
    :return:
    """

    # Class IDs per ROI
    class_ids = tf.argmax(roi_scores_softmax, axis=1, output_type=tf.int32)

    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(roi_scores_softmax.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(roi_scores_softmax, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(roi_txtytwth, indices)
    # Apply bounding box deltas
    # Shape: [num_rois, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = decode_bbox_with_mean_and_std(rois, deltas_specific,
                                                 target_means, target_stds)
    refined_rois, refined_rois_idx = bboxes_clip_filter_tf(refined_rois, 0, image_shape[0], image_shape[1],
                                                           min_edge=None)
    # TODO: remove min edge

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]

    # Filter out low confidence boxes
    score_keep = tf.where(class_scores >= score_threshold)[:, 0]
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(score_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=max_num_per_class,
            iou_threshold=nms_iou_threshold)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        tf.logging.debug('nms keep map is {}'.format(class_keep))
        return class_keep

    # 2. Map over class IDs
    nms_keep = []
    for i in range(unique_pre_nms_class_ids.shape[0]):
        nms_keep.append(nms_keep_map(unique_pre_nms_class_ids[i]))

    if len(nms_keep) == 0:
        return None, None, None
    nms_keep = tf.concat(nms_keep, axis=0)

    # 3. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = max_num_per_image
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    return tf.gather(refined_rois, keep), tf.gather(class_ids, keep), tf.gather(class_scores, keep)


def post_ops_prediction(roi_scores_softmax, roi_txtytwth, rois, image_shape,
                        target_means, target_stds,
                        max_num_per_class=15,
                        max_num_per_image=150,
                        nms_iou_threshold=0.3,
                        score_threshold=0.05,
                        extractor_stride=16,
                        num_classes=21,
                        ):
    """

    :param roi_scores_softmax:      [num_rois, num_classes]
    :param roi_txtytwth:            [num_rois, num_classes, 4]
    :param rois:                    [num_rois, 4]
    :param image_shape:             [2,]
    :param target_means:            [4,]
    :param target_stds:             [4,]
    :param max_num_per_class:
    :param max_num_per_image:
    :param nms_iou_threshold:
    :param score_threshold:
    :param extractor_stride:
    :param num_classes:
    :return:
    """
    if target_stds is None:
        target_stds = [1, 1, 1, 1]
    if target_means is None:
        target_means = [0, 0, 0, 0]
    res_scores = []
    res_bboxes = []
    res_cls = []
    for i in range(1, num_classes):
        inds = tf.where(roi_scores_softmax[:, i] > score_threshold)[:, 0]
        cls_score = tf.gather(roi_scores_softmax[:, i], inds)
        # cls_score = roi_scores_softmax[inds, i]
        final_bboxes = decode_bbox_with_mean_and_std(rois, roi_txtytwth[:, i, :],
                                                     target_means, target_stds)
        final_bboxes, _ = bboxes_clip_filter_tf(final_bboxes, 0, image_shape[0], image_shape[1])
        final_bboxes = tf.gather(final_bboxes, inds)

        keep = tf.image.non_max_suppression(final_bboxes, cls_score, max_num_per_class, iou_threshold=nms_iou_threshold)
        if tf.size(keep).numpy() == 0:
            continue
        res_scores.append(tf.gather(cls_score, keep))
        res_bboxes.append(tf.gather(final_bboxes, keep))
        res_cls.append(tf.ones_like(keep, dtype=tf.int32) * i)

    if len(res_scores) == 0:
        return None, None, None

    scores_after_nms = tf.concat(res_scores, axis=0)
    bboxes_after_nms = tf.concat(res_bboxes, axis=0)
    cls_after_nms = tf.concat(res_cls, axis=0)

    _, final_idx = tf.nn.top_k(scores_after_nms, k=tf.minimum(max_num_per_image, tf.size(scores_after_nms)),
                               sorted=False)
    return tf.gather(bboxes_after_nms, final_idx), tf.gather(cls_after_nms, final_idx), tf.gather(scores_after_nms,
                                                                                                  final_idx)
