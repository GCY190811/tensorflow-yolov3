#! /usr/bin/env python
# coding=utf-8

import sys
import argparse
import numpy as np
import tensorflow as tf
import core.high_level_utils as hl_utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_txt", help="train or valid file")
    parser.add_argument("tfrecord_path_prefix",
                        help='path and tfrecord prefix')
    parser.add_argument("num_tfrecords", default=50, type=int)
    flags = parser.parse_args()

    dataset = hl_utils.read_image_box_from_text(
        flags.dataset_txt)  # {imagePath: (box, label)}
    image_paths = list(dataset.keys())
    images_num = len(image_paths)
    print("==>> Processing %d images" % images_num)
    per_tfrecord_images = images_num // flags.num_tfrecords

    n = 0
    while n <= flags.num_tfrecords:
        tfrecord_file = flags.tfrecord_path_prefix + "%04d.tfrecords" % n
        with tf.python_io.TFRecordWriter(tfrecord_file) as record_writer:
            st = n * per_tfrecord_images
            en = (n +
                  1) * per_tfrecord_images if n < flags.num_tfrecords else len(
                      image_paths)
            for i in range(st, en):
                image = tf.gfile.GFile(image_paths[i], 'rb').read()
                bboxes, labels = dataset[image_paths[i]]
                bboxes = bboxes.tostring()

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image':
                        tf.train.Feature(bytes_list=tf.train.BytesList(
                            value=[image])),
                        'bboxes':
                        tf.train.Feature(bytes_list=tf.train.BytesList(
                            value=[bboxes])),
                        'labels':
                        tf.train.Feature(int64_list=tf.train.Int64List(
                            value=labels)),
                    }))

                record_writer.write(example.SerializeToString())
            print(">> Saving %5d images in %s" % (en - st, tfrecord_file))
            n += 1


if __name__ == "__main__": main(sys.argv[1:])
