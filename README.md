# TF EAGER OBJECT DETECTION

## 0. Targets
+ TensorFlow Eager Mode.
+ Object detection models.

## 1. Architecture
+ `scripts`:
    + `generate_pascal_tf_records.py`: generate tfrecords files from pascal source files.
    + `train.py`: train coco or pascal.
    + `eval_pascal.py`: eval pascal dataset.
    + `label_map_src`: copy from TensorFlow Object Detection API.
+ `object_detection/dataset`:
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
        + `resnet_faster_rcnn.py`: resnet faster rcnn model.
    + `fpn`:
        + `base_fpn_model.py`: base class for fpn.
        + `resnet_fpn.py`: resnet fpn model.
    + `model_factory`: factory for model creation.
    + `anchor_target.py`: generate anchor target for rpn training.
    + `losses.py`: smooth l1 loss & cross entropy loss.
    + `prediction.py`: generate predictions after roi head.
    + `proposal_target.py`: generate proposal target for roi training.
    + `region_proposal.py`: generate region proposals for both training & testing procedure.
    + `roi_pooling.py`: roi pooling results.
+ `object_detection/protos`: protobuf source files.
    + `protoc ./object_detection/protos/*.proto --python_out=./object_detection/protos/ `
+ `object_detection/utils`:
    + `anchor_generator.py`: generate anchors.
    + `bbox_np.py`: cal iou, bbox range filter and bbox clip filter by np.
    + `bbox_tf.py`: cal iou, bbox range filter and bbox clip filter by tf.
    + `bbox_transform.py`: convert between bbox(xmin, ymin, xmax, ymax) and pred(tx, ty, tw, th)
    + `visual_utils.py`: draw bboxes in an image.
    + `pytorch_to_tf.py`: convert pytorch model to pickle map.


---


## 2. TODO

### 2.1. dataset
+ [x] pascal training dataset.
+ [x] pascal evaluating dataset.
+ [x] coco training dataset.
+ [ ] coco evaluating dataset.

### 2.2. model
+ [x] faster rcnn
+ [x] fpn
+ [ ] mask rcnn

### 2.3. training & evaluating
+ [ ] use `defun` in all components.
+ [ ] multi gpu support.

### 2.4. others
+ [ ] BUG: after a few epochs, gpu memory will boomed twice...
+ [ ] jupyter samples.

---

## 3. training records
+ trained on voc2007 trainval, evaluating on pascal 2007 test set(use voc2007 metric).

### 3.1. VGG16-Faster-RCNN
| Models | mAP |
|:------:|:-----:|
|tf-faster-rcnn(source)|0.708|
|tf-faster-rcnn(load pre-trained model)|0.7106|
|typical configs|0.6935/0.6869|
|without data argument|0.6659|

### 3.2. ResNet-Faster-RCNN
+ resnet 101

| Models | mAP |
|:------:|:-----:|
|tf-faster-rcnn(source)|0.757|
|tf-faster-rcnn(load pre-trained model)|0.7578|
|typical configs|0.7456/0.7296|
|remove bias for extractor|0.7307/0.7270|

### 3.3. FPN-Tensorflow
+ resnet 50

| Models | mAP |
|:------:|:-----:|
|FPN_Tensorflow(source)|0.7426|
|tf-faster-rcnn(load pre-trained model)|0.7430|
|14 epochs|0.70|


---

## 4. 可有可无的教程……
+ training on pascal voc 2007 trainval set, evaluating on pascal voc 2007 test set.
+ Step 0: generate python protos by `protoc ./object_detection/protos/*.proto --python_out=./object_detection/protos/ `.
+ Step 1: generate trainval datasets, set configs and use `python scripts/generate_pascal_tf_records.py`.
+ Step 2: training by `python scripts/train.py`, get logs at `/path/to/logs_dir/`.
+ Step 3: evaluating by `python scripts/eval_pascal.py /path/to/logs_dir/ckpt`.