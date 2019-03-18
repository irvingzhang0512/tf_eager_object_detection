# TF EAGER OBJECT DETECTION

## 0. Targets
+ TensorFlow Eager Mode.
+ Object Detection Model.
    + [x] faster rcnn
    + [ ] fpn
    + [ ] mask rcnn


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
+ [x] add resnet faster rcnn model.
+ [ ] eval result file paths.
+ [ ] implement proposal target with tf version of `numpy.random.choice`.


## 3. training records

### 3.1. VGG16-Faster-RCNN
+ load tf-faster-rcnn pre-trained model，mAP of pascal 2007 test set is 0.71。
+ end-to-end training：load slim pretrained model (`logs-pascal-slim`):
    + SGD，1e-3 -> 1e-4: after 14 epochs, mAP is 0.6935(or 0.6869).
    + SGD，1e-3 -> 1e-4: without data argument: after 14 epochs, mAP is 0.6659.
+ alt training：load slim pretrained model
    + Step 1: training extractor & rpn head(rpn loss, 14 epochs, no l2 loss, `logs-pascal-slim-rpn`)
    + Step 2: training roi head & roi pooling(roi loss, 14 epochs, `logs-pascal-slim-roi-after-rpn`)
    + Step 3: training rpn head only(rpn loss, 14 epochs, `logs-pascal-slim-rpn-only`), mAP is 0.6683.
    + Step 4: training roi head only(roi loss, 14 epochs, `logs-pascal-slim-roi-only`), mAP is 0.6733.

### 3.2. ResNet-Faster-RCNN
+ resnet 101
    + load tf-faster-rcnn pre-trained model, mAP of pascal 2007 test set:
        + standard: 0.7566
        + set height & width to 32x: 0.7218
    + end to end training: load keras pre-trained model(`logs-pascal-resnet101-default`):
        + without roi pooling max pooling(wrong bn trainable): mAP 0.41, 0.49, 0.54, 0.559, 0.568, 0.61(14 epochs).
        + without roi pooling max pooling: mAP 0.408, 0.4842
        + with roi pooling max pooling(wrong bn trainable): mAP 0.6434(11 epoch), 0.6468, 0.6423, 0.6351

## 4. 可有可无的教程……
+ training on pascal voc 2007 trainval set, evaluating on pascal voc 2007 test set.
+ Step 1: generate trainval datasets, set configs and use `python scripts/generate_pascal_tf_records.py` to generate tf records.
+ Step 2: training by `python scripts/train.py`, get logs at `/path/to/logs_dir/`.
+ Step 3: evaluating by `python scripts/eval_pascal.py /path/to/logs_dir/ckpt`.