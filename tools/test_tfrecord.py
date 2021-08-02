import cv2
import numpy as np
import tensorflow as tf


def test_normal():
    filename = '/tmp/images.tfrecords'
    dataset = tf.data.TFRecordDataset(filename)
    for tf_example in dataset:
        image_feature_description = {
            "bbox": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
        }
        x = tf.io.parse_example(tf_example, image_feature_description)
        image = tf.io.decode_image(x['image'], channels=3)
        height = int(x['height'])
        width = int(x['width'])
        image = tf.reshape(image, (height, width, 3))
        image = image.numpy()

        origin_height = tf.cast(x['height'], tf.float32)
        origin_width = tf.cast(x['width'], tf.float32)

        # Draw bboxes
        bboxes = tf.cast(
            tf.io.decode_raw(x["bbox"], out_type=tf.int64), dtype=tf.float32
        )
        bboxes = tf.reshape(bboxes, (-1, 4)).numpy()
        labels = tf.io.decode_raw(x["label"], out_type=tf.int64).numpy()

        print(bboxes, labels)

        classes = [
            'person',
            'hardhat',
            'glasses',
            'vest',
            'gloves',
            'hand',
        ]
        class_map = dict(list(enumerate(classes)))
        colors = [tuple(map(int, np.random.randint(0, 256, 3))) for _ in range(len(labels))]
        for bbox, label in zip(bboxes, labels):
            bbox = list(map(int, bbox))
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            color = colors[label-1]
            class_name = class_map[label-1]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)
        cv2.imshow('img', image[:, :, ::-1])
        cv2.waitKey(0)
        break


def test_preprocessing():
    pass


if __name__ == '__main__':
    test_normal()