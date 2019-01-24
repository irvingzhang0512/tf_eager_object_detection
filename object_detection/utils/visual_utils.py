import tensorflow as tf
import cv2


def draw_bboxes_with_labels(image, bboxes, label_texts):
    """
    在ndarray或tf.Tensor对象上，画bboxes和对应的labels
    :param image:       一张图片，shape 为 [height, width, channels]
    :param bboxes:      一组bounding box，shape 为 [bbox_number, 4]，顺序为 ymin, xmin, ymax, xmax
                        float类型，取值范围[0, 1]
    :param label_texts:      要显示的标签，shape为(bbox_number, )
    :return:        画完bbox的图片，为ndarray类型，shape与输入相同
    """
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    height, width, channels = image.shape
    for bbox, cur_label in zip(bboxes, label_texts):
        ymin, xmin, ymax, xmax = int(bbox[0] * height), int(bbox[1] * width), int(bbox[2] * height), int(
            bbox[3] * width)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img=image,
                    text=str(cur_label.numpy()),
                    org=(xmin, ymin + 10),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1e-3 * image.shape[0],
                    color=(0, 0, 255),
                    thickness=2
                    )
    return image
