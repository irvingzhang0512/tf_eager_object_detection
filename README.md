# TF EAGER OBJECT DETECTION

## 0. Targets
+ 小目标……实现中……
+ TensorFlow Eager Mode.
+ Object Detection Model.
    + faster rcnn -> mask rcnn
    + ssd


## 1. Architecture
+ `alt_training.py`: alternative training scripts.
+ `end2end_training.py`: end to end training scripts.
+ `scripts`:
    + `generate_pascal_tf_records.py`: generate tfrecords files from pascal source files.
+ `object_detection/model`:
    + `faster_rcnn_config.py`: configs using in faster rcnn.
+ `object_detection/dataset`:
    + `pascal_tf_dataset_generator`: 
        + get `tf.data.Dataset` object from tfrecords files.
        + use `imgaug` for data argumentation.
    + `label_map_utils.py`: copy from TensorFlow Object Detection API.
    + `tf_record_utils.py`: utils to generate tfrecords files.    
+ `object_detection/model`:
    + `extractor`: get feature maps from images(after preprocessing and data argument).
        + `resnet.py`: copy from tf eager fasterrcnn.
        + `feature_extractor.py`: copy from `tf.keras.applications`.
    + `faster_rcnn.py`: base model for faster rcnn. 
        + end2end training: `BaseRoiModel`, `BaseRpnModel`, `RpnTrainingModel`, `RoiTrainingModel`
        + alternative training: `BaseFasterRcnnModel`, `FasterRcnnTrainingModel`
        + prediction: `post_ops_prediction`.
    + `rpn.py`: rpn head & training proposal & proposal.
    + `roi.py`: roi head & pooling & training proposal.
    + `losses.py`: cal class loss & reg loss for rpn training and roi training.
+ `object_detection/protos`: protobuf source files.
    + `protoc ./object_detection/protos/*.proto --python_out=./object_detection/protos/ `
+ `object_detection/utils`:
    + `anchors.py`: generate anchors.
    + `bbox_np.py`: cal iou by np.
    + `bbox_tf.py`: cal iou by tf.
    + `bbox_transform.py`: convert between bbox(ymin, xmin, ymax, xmax) and pred(tx, ty, tw, th)
    + `pascal_voc_map_utils`: cal pascal metrics mAP.
    + `visual_utils.py`: draw bboxes in an image.

## 2. TODO
+ [x] use different preprocessing utils for different feature extractor.
+ [x] remove all magic number and use config dict to cover all params in faster rcnn.
+ [x] add l2 regularize loss.
+ [x] compare current net with the original faster rcnn model.
+ [x] add summaries in training procedure.
+ [x] alternative training.
+ [x] use `logging` instead of `print`.
+ [x] COCO dataset & training.
+ [x] set bboxes range in `[0, height - 1]` & `[0, width - 1]`.
+ [x] add prediction_score_threshold for image summary.
+ [x] add model load/save functions.
+ [ ] predict and visual scripts.


## 3. 训练记录

### 3.1. end to end training:
+ 整体导入 tf-faster-rcnn 模型(`logs-pascal-rpn-roi`)
    + 直接预测，map结果为0.71。
    + 训练1个epoch后map为0.71016，SGD + lr 1e-4。
    + 训练14个epoch后map为0.7051，SGD + lr 1e-4 -> 1e-5。
+ extractor & rpn head 使用 tf-faster-rcnn　的模型，只训练 roi head(`logs-pascal-roi`)
+ 使用 slim pretrained model 训练整体模型（`logs-pascal-slim`）：
    + SGD，1e-3 -> 1e-4：1个epoch后map为0.42，14个epoch后map为（Terminal 2）：

### 3.2. alt training
+ 使用 slim pretrained model，只训练 extractor & rpn head（14个epoch，无l2 loss，`logs-pascal-slim-rpn`），之后：
    + 训练 roi head（14个epoch，`logs-pascal-slim-roi-after-rpn`，l2 loss，sgd, 1e-3 -> 1e-4，Terminal 1）：
    + 训练 roi head（14个epoch，无l2 loss，adam, 1e-4 -> 1e-5）：
+ 使用 slim pretrained model，只训练 extractor & rpn head（14个epoch，有l2 loss，`logs-pascal-slim-rpn-l2`），之后：
    + 训练 roi head（14个epoch，`logs-pascal-slim-roi-after-rpn-l2`，sgd, 1e-3 -> 1e-4）：
    + 训练 roi head（14个epoch，l2 loss，adam, 1e-4 -> 1e-5）：
    