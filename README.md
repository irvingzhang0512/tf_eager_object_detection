# TF EAGER OBJECT DETECTION

## 0. Targets
+ 小目标……实现中……
+ TensorFlow Eager Mode.
+ Object Detection Model.
    + faster rcnn -> mask rcnn
    + ssd


## 1. Architecture
+ `scripts`:
    + `generate_pascal_tf_records.py`: generate tfrecords files from pascal source files.
    + `train.py`: train coco or pascal.
    + `eval_pascal.py`: eval 2007 pascal test set.
+ `object_detection/model`:
    + `faster_rcnn_config.py`: configs of faster rcnn.
+ `object_detection/dataset`:
    + `label_map_src`: copy from TensorFlow Object Detection API.
    + `utils`:
        + `label_map_utils.py`: copy from TensorFlow Object Detection API.
        + `tf_record_utils.py`: utils to generate tfrecords files.    
        + `tf_dataset_utils.py`: utils to generate `tf.data.Dataset` objects.
    + `pascal_tf_dataset_generator.py`: get training pascal `tf.data.Dataset` object from tfrecords files.
    + `pascal_tf_dataset_local_file.py`: get training pascal `tf.data.Dataset` by local files.
    + `coco_tf_dataset_generator.py`: get training coco `tf.data.Dataset` object.
    + `eval_pascal_tf_dataset.py`: get eval pascal `tf.data.Dataset` object.
+ `object_detection/evaluation`:
    + `detectron_pascal_evaluation_utils.py`: copy from `Detectron`, eval pascal with local detection results.
    + `pascal_eval_files_utils.py`: generate local detection result files.
    + `pascal_voc_map_utils.py`: get pascal map results.
+ `object_detection/model`:
    + `faster_rcnn`:
        + `base_faster_rcnn_model.py`: base class for faster rcnn.
        + `vgg16_faster_rcnn.py`: vgg16 faster rcnn model.
    + `anchor_target.py`: generate anchor target for rpn training.
    + `losses.py`: smooth l1 loss & cross entropy loss.
    + `prediction.py`: generate predictions after roi head.
    + `proposal_target.py`: generate proposal target for roi training.
    + `region_proposal.py`: generate region proposals for both training & testing procedure.
    + `roi_pooling`: roi pooling results.
+ `object_detection/protos`: protobuf source files.
    + `protoc ./object_detection/protos/*.proto --python_out=./object_detection/protos/ `
+ `object_detection/utils`:
    + `anchors.py`: generate anchors.
    + `bbox_np.py`: cal iou, bbox range filter and bbox clip filter by np.
    + `bbox_tf.py`: cal iou, bbox range filter and bbox clip filter by tf.
    + `bbox_transform.py`: convert between bbox(xmin, ymin, xmax, ymax) and pred(tx, ty, tw, th)
    + `visual_utils.py`: draw bboxes in an image.
    + `pytorch_to_tf.py`: convert pytorch model to pickle map.

## 2. TODO
+ [x] use different preprocessing utils for different feature extractor.
+ [x] remove all magic number and use config dict to cover all params in faster rcnn.
+ [x] add l2 regularize loss.
+ [x] compare current net with the original faster rcnn model.
+ [x] add summaries in training procedure.
+ [x] use `logging` instead of `print`.
+ [x] COCO dataset & training.
+ [x] set bboxes range in `[0, height - 1]` & `[0, width - 1]`.
+ [x] add prediction_score_threshold for image summary.
+ [x] add model load/save functions.
+ [x] predict and visual scripts.
+ [ ] add resnet faster rcnn model.


## 3. 训练记录

+ 整体导入 tf-faster-rcnn 模型，直接预测，map结果为0.71。
+ end-to-end training：使用 slim pretrained model 训练整体模型（`logs-pascal-slim`）：
    + SGD，1e-3 -> 1e-4：14个epoch后map为0.6935（0.6869）。
    + SGD，1e-3 -> 1e-4，输入数据无镜像增强：14个epoch后map为0.6659。
+ alt training：使用 slim pretrained model
    + 第一步：只训练 extractor & rpn head（14个epoch，无l2 loss，`logs-pascal-slim-rpn`）。
    + 第二步：训练 roi head & roi pooling（14个epoch，`logs-pascal-slim-roi-after-rpn`）。
    + 第三步：只训练 rpn head（14个epoch，`logs-pascal-slim-rpn-only`），map为0.6683。
    + 第四步：只训练 roi head（14个epoch，`logs-pascal-slim-roi-only`），map为0.6733。
    