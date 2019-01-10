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
            + rpn head
            + rpn proposal
            + rpn training sampler & losses
        + roi.
            + roi pooling
            + roi head
            + roi training sampler & losses
        + prediction.
            + generate final prediction results.
        + anchor generator.
            + base generator.
            + anchor filter(TODO)
        + bbox ops.
            + (anchors, gt_bboxes) -> (tx ty tw th)
            + (anchors, bboxes_txtytwth) -> (target bboxes)
            + cal iou.
    + ssd
+ train
+ predict