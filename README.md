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


## 1. TODO
+ dataset:
    + generate tfrecords file from pascal source files.
    + get `tf.data.Dataset` object from tfrecords files.
    + use `imgaug` for data argumentation.
+ model:
    + faster rcnn.
        + feature extractor.
        + rpn.
        + roi pooling.
        + anchor generator.
        + bbox transformations.
        + ground truth generator.
        + bbox & classes classifier by rois and shared features.
    + ssd
+ train
+ predict