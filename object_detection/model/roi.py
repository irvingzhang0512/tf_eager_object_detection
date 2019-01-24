import tensorflow as tf
from object_detection.utils.bbox_tf import pairwise_iou
from object_detection.model.losses import cls_loss, smooth_l1_loss
from object_detection.utils.bbox_transform import encode_bbox

layers = tf.keras.layers


class RoiPooling(tf.keras.Model):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size
        self._concat_layer = layers.Concatenate(axis=0)
        self._flatten_layer = layers.Flatten()

    def call(self, inputs, training=None, mask=None):
        """
        输入 backbone 的结果和 rpn proposals 的结果(即 RPNProposal 的输出)
        输出 roi pooloing 的结果，即在特征图上，对每个rpn proposal获取一个固定尺寸的特征图
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        # [1, height, width, channels]  [num_rois, 4]
        shared_layers, rois = inputs

        # TODO: ROI Polling 的细节
        res = []
        for roi in rois:
            ymin = tf.to_int32(roi[0])
            xmin = tf.to_int32(roi[1])
            ymax = tf.to_int32(roi[2])
            xmax = tf.to_int32(roi[3])
            res.append(
                tf.image.resize_bilinear(shared_layers[:, ymin:ymax + 1, xmin:xmax + 1, :],
                                         [self._pool_size, self._pool_size]))

        net = self._concat_layer(res)
        print(net.shape)
        return self._flatten_layer(net)


class RoiHead(tf.keras.Model):
    def __init__(self, num_classes, ):
        super().__init__()
        self._num_classes = num_classes

        # TODO: Dense 层的细节
        self._fc1 = layers.Dense(4096)
        self._fc2 = layers.Dense(4096)

        self._score_prediction = layers.Dense(num_classes)
        self._bbox_prediction = layers.Dense(4)

    def call(self, inputs, training=None, mask=None):
        """
        输入 roi pooling 的结果
        对每个 roi pooling 的结果进行预测（预测bboxes）
        :param inputs:  roi_features, [num_rois, len_roi_feature]
        :param training:
        :param mask:
        :return:
        """
        net = self._fc1(inputs)
        net = self._fc2(net)
        roi_score = self._score_prediction(net)
        roi_bboxes_txtytwth = self._bbox_prediction(net)

        return roi_score, roi_bboxes_txtytwth


class RoiTrainingProposal(tf.keras.Model):
    def __init__(self,
                 pos_iou_threshold=0.5,
                 neg_iou_threshold=0.1,
                 total_num_samples=128,
                 max_pos_samples=32, ):
        super().__init__()

        self._pos_iou_threshold = pos_iou_threshold
        self._neg_iou_threshold = neg_iou_threshold
        self._total_num_samples = total_num_samples
        self._max_pos_samples = max_pos_samples

    def call(self, inputs, training=None, mask=None):
        """
        生成训练roi用的数据
        总体过程：
        1. 计算 rois 与gt_bbox（即输入数据中的bbox）的iou。
        2. 设置（iou > 0.7）或与每个gt_bbox iou最大的anchor为整理，数量最多为
        3. 设置 iou < 0.3 的anchor为反例。
        4. 设置其他anchor不参与训练。
        5. 返回参与训练anchor的ids，以及这些ids根据顺序排列的label列表
        6. 对于正例，还需要返回其对应的 gt_bbox 的编号
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        rois, gt_bboxes = inputs

        # [rois_size, gt_bboxes_size]
        labels = -tf.ones((rois.shape[0],), tf.int32)
        iou = pairwise_iou(rois, gt_bboxes)

        # 设置与任意gt_bbox的iou > pos_iou_threshold 的 roi 为正例
        # 设置与所有gt_bbox的iou < neg_iou_threshold 的 roi 为反例
        max_scores = tf.reduce_max(iou, axis=1)
        gt_bbox_idx = tf.argmax(iou, axis=1)
        labels = tf.where(max_scores >= self._pos_iou_threshold, tf.ones_like(labels), labels)
        labels = tf.where(max_scores <= self._neg_iou_threshold, tf.zeros_like(labels), labels)

        # 计算正反例真实数量
        total_pos_num = tf.reduce_sum(tf.where(tf.equal(labels, 1), tf.ones_like(labels), tf.zeros_like(labels)))
        total_neg_num = tf.reduce_sum(tf.where(tf.equal(labels, 0), tf.ones_like(labels), tf.zeros_like(labels)))

        # 根据要求，修正正反例数量
        cur_pos_num = tf.minimum(total_pos_num, self._max_pos_samples)
        cur_neg_num = tf.minimum(self._total_num_samples - cur_pos_num, total_neg_num)
        print('roi training has %d pos samples and %d neg samples' % (cur_pos_num, cur_neg_num))

        # 随机选择正例和反例
        _, total_pos_index = tf.nn.top_k(max_scores, total_pos_num, sorted=False)
        _, total_neg_index = tf.nn.top_k(-max_scores, total_neg_num, sorted=False)
        pos_index = tf.gather(total_pos_index, tf.random_shuffle(tf.range(0, total_pos_num))[:cur_pos_num])
        neg_index = tf.gather(total_neg_index, tf.random_shuffle(tf.range(0, total_neg_num))[:cur_neg_num])

        # 生成最终结果
        selected_idx = tf.concat([pos_index, neg_index], axis=0)
        return selected_idx, tf.gather(labels, selected_idx), tf.gather(gt_bbox_idx, selected_idx)


class RoiTrainingModel(tf.keras.Model):
    def __init__(self,
                 cls_loss_weight=1.0,
                 reg_loss_weight=2.0, ):
        super().__init__()
        self._cls_loss_weight = cls_loss_weight
        self._reg_loss_weight = reg_loss_weight

        self._roi_training_proposal = RoiTrainingProposal()

    def call(self, inputs, training=None, mask=None):
        rpn_proposals_bboxes, roi_score, roi_bboxes_txtytwth, gt_bboxes, gt_labels = inputs
        gt_labels = tf.to_int32(gt_labels)

        # ROI 训练相关
        # 获取 roi 的训练数据
        roi_training_idx, roi_training_labels, roi_training_gt_bbox_idx = self._roi_training_proposal(
            (rpn_proposals_bboxes,
             gt_bboxes), training, mask)

        # cal roi cls loss
        roi_training_score = tf.gather(roi_score, roi_training_idx)
        roi_cls_loss = cls_loss(logits=roi_training_score,
                                labels=tf.gather(gt_labels, roi_training_gt_bbox_idx) * tf.to_int32(
                                    roi_training_labels))

        # cal roi bbox reg loss
        # inputs: roi_training_idx, roi_training_labels, roi_training_gt_bbox_idx,
        #         roi_predicting_bboxes, rpn_proposals_bboxes, gt_bboxes
        # 只计算正例的损失函数
        selected_rpn_proposal_bboxes = tf.gather(rpn_proposals_bboxes, roi_training_idx)
        selected_gt_bboxes = tf.gather(gt_bboxes, roi_training_gt_bbox_idx)
        bboxes_txtytwth_gt = encode_bbox(selected_rpn_proposal_bboxes.numpy(),
                                         selected_gt_bboxes.numpy())
        bboxes_txtytwth_pred = tf.gather(roi_bboxes_txtytwth, roi_training_idx)
        roi_reg_loss = smooth_l1_loss(bboxes_txtytwth_pred, bboxes_txtytwth_gt,
                                      inside_weights=roi_training_labels,
                                      outside_weights=self._reg_loss_weight)

        return roi_cls_loss, roi_reg_loss
