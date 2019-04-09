# Dataset Module
+ target: generate `tf.data.Dataset` object for object detection tasks.
+ classification:
    + training dataset
    + eval dataset
    + utils

---

## 1. training dataset

### 1.1. iter features
+ every iter for each training set generate 3 features: preprocessed image, bboxes and labels.
+ preprocessed image: 
    + dtype: `tf.float32`
    + shape: `[1, None, None, 3]`
    + PS: `bgr` format.
+ bboxes: 
    + dytpe: `tf.float32`
    + shape: `[1, None, 4]`
    + format: `ymin, xmin, ymax, xmax`
    + range: `[0, image_height - 1]` or `[0, image_width - 1]`
+ labels:
    + dtype: `tf.int32` or `tf.int64`
    + shape: `[1, None,]`

### 1.2. data flow
+ input: rgb uint8 raw image.
+ data argument:
    + random flip left and right.
+ resize image with min_edge and max_edge.
+ preprocessing(one of the following methods):
    + method 1(caffe): convert 'rgb' to 'bgr', and then subject imagenet means.
    + method 2(tf): convert `[0, 255]` to `[-1, 1]`

---

## 2. eval dataset

### 2.1. iter features
+ every iter for each training set generate 3 features: preprocessed image, image scale , image raw height and image raw width.
+ preprocessed image: 
    + dtype: `tf.float32`
    + shape: `[1, None, None, 3]`
    + PS: `bgr` or `rgb`
+ image scale: 
    + dytpe: `tf.float64`
    + shape: `[1,]`
+ image height:
    + dtype: `tf.int32` or `tf.int64`
    + shape: `[1,]`
+ image width:
    + dtype: `tf.int32` or `tf.int64`
    + shape: `[1,]`
+ image_id:
    + dtype: `tf.float32`
    + shape: `[1,]`
    + PS: for COCO dataset only. used in coco eval tools.

### 2.2. data flow
+ input: rgb uint8 raw image.
+ resize image with min_edge and max_edge.
+ preprocessing(one of the following methods):
    + method 1(caffe): convert 'rgb' to 'bgr', and then subject imagenet means.
    + method 2(tf): convert `[0, 255]` to `[-1, 1]`
+ convert `rgb` to `bgr` if necessary.
