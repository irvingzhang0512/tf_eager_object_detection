```
/**
 *                             _ooOoo_
 *                            o8888888o
 *                            88" . "88
 *                            (| -_- |)
 *                            O\  =  /O
 *                         ____/`---'\____
 *                       .'  \\|     |//  `.
 *                      /  \\|||  :  |||//  \
 *                     /  _||||| -:- |||||-  \
 *                     |   | \\\  -  /// |   |
 *                     | \_|  ''\---/''  |   |
 *                     \  .-\__  `-`  ___/-. /
 *                   ___`. .'  /--.--\  `. . __
 *                ."" '<  `.___\_<|>_/___.'  >'"".
 *               | | :  `- \`.;`\ _ /`;.`/ - ` : | |
 *               \  \ `-.   \_ __\ /__ _/   .-` /  /
 *          ======`-.____`-.___\_____/___.-`____.-'======
 *                             `=---='
 *          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
*/
```

## 0. Targets
+ 小目标……实现中……
+ TensorFlow Eager Mode.
+ Object Detection Model.
    + faster rcnn -> mask rcnn
    + ssd


## 1. Architecture
+ `scripts`:
    + `generate_pascal_tf_records.py`: generate tfrecords file from pascal source files.
+ `object_detection/dataset`:
    + `pascal_tf_dataset_generator`: 
        + get `tf.data.Dataset` object from tfrecords files.
        + use `imgaug` for default data argumentation.
    + `label_map_utils.py`: copy from TensorFlow Object Detection API.
    + `tf_record_utils.py`: utils to generate tfrecords files.    
+ `object_detection/model`:
    + `feature_extractor.py`: get shared features from raw image, usually use VGG16 & ResNet.
        + copy codes from keras source codes.
    + `faster_rcnn.py`: base model for faster rcnn. get shared features & rpn scores and rpn bboxes.
        + has a `feature_extractor` implementing in `feature_extractor.py`.
    + `rpn.py`: rpn training model & rpn proposal model.
    + `roi.py`: roi training model.
    + `losses.py`: cal class loss & reg loss for rpn training and roi training.

## 2. TODO
+ [x] use different preprocessing utils for different feature extractor.
+ [x] remove all magic number and use config dict to cover all params in faster rcnn.
+ [x] add l2 regularize loss.
+ [x] compare current net with the original faster rcnn model.
+ [x] add summaries in training procedure.
+ [x] alternative training.
+ [ ] add model load/save functions.
+ [ ] predict and visual scripts.
+ [ ] use `logging` instead of `print`.


## 3. 存在的问题
`protoc ./object_detection/protos/*.proto --python_out=./object_detection/protos/ `


