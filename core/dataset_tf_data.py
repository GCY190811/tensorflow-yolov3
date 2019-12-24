"""
use tf.data API
"""

import random
import tensorflow as tf
from core.config import cfg
import helper.image as img
import numpy as np


def data_func(trainPattern, valPattern):
    train_dataset = tf.data.TFRecordDataset(
        filenames=tf.gfile.Glob(trainPattern))
    train_dataset = train_dataset.map(parse_func("train"),
                                      num_parallel_calls=6)
    # repeat 如何处理
    train_dataset = train_dataset.repeat(cfg.TRAIN.WARMUP_EPOCHS +
                                         cfg.TRAIN.FIRST_STAGE_EPOCHS +
                                         cfg.TRAIN.SECOND_STAGE_EPOCHS)

    train_dataset = train_dataset.shuffle(cfg.SHUFFLE_SIZE)
    train_dataset = train_dataset.batch(cfg.TRAIN.BATCH_SIZE).prefetch(1)

    val_dataset = tf.data.TFRecordDataset(filenames=tf.gfile.Glob(valPattern))
    val_dataset = val_dataset.map(parse_func, num_parallel_calls=6)
    # repeat 如何处理
    val_dataset = val_dataset.repeat(cfg.TRAIN.WARMUP_EPOCHS +
                                     cfg.TRAIN.FIRST_STAGE_EPOCHS +
                                     cfg.TRAIN.SECOND_STAGE_EPOCHS)
    val_dataset = val_dataset.shuffle(cfg.SHUFFLE_SIZE)
    val_dataset = val_dataset.batch(cfg.TRAIN.BATCH_SIZE).prefetch(1)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)

    return train_init_op, val_init_op, iterator.get_next()


def parse_func(mode, serialized_example):
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image':
                                           tf.FixedLenFeature([], tf.string),
                                           'bboxes':
                                           tf.FixedLenFeature([], tf.strin),
                                           'labels':
                                           tf.FixedLenFeature(dtype=tf.int64),
                                       })

    image = tf.image.decode_jpeg(features['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    true_boxes = tf.decode_raw(features['bboxes'], tf.float32)
    true_boxes = tf.reshape(true_boxes, shape=[-1, 4])

    true_labels = features['labels'].values

    return tf.py_func(preprocess, inp=[], Tout=[])


def preprocess(mode, image, true_boxes, true_labels):
    input_sizes = cfg.TRAIN.INPUT_SIZE if mode == 'train' else cfg.TEST.INPUT_SIZE
    data_aug = cfg.TRAIN.DATA_AUG if mode == 'train' else cfg.TEST.DATA_AUG

    strides = np.array(cfg.YOLO.STRIDES)  # downsample ratio [8, 16, 32]
    classes = utils.read_class_names(cfg.YOLO.CLASSES)
    num_classes = len(classes)
    anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))  # numpy (3,3,2)
    anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
    max_bbox_per_scale = cfg.YOLO.MAX_BBOX_PER_SCALE

    input_size = random.choice(input_sizes)
    output_sizes = input_size // strides  # 3 output layer

    image, bboxes = image_preprocess(image, true_boxes, input_size)
    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = bboxes_preprocess(
        bboxes, true_labels, output_sizes, num_classes)

    batch_image = np.zeros((input_size, input_size, 3))
    batch_label_sbbox = np.zeros(
        (output_sizes[0], output_sizes[0], anchor_per_scale, 5 + num_classes))
    batch_label_mbbox = np.zeros(
        (output_sizes[1], output_sizes[1], anchor_per_scale, 5 + num_classes))
    batch_label_lbbox = np.zeros(
        (output_sizes[2], output_sizes[2], anchor_per_scale, 5 + num_classes))
    batch_sbboxes = np.zeros((max_bbox_per_scale, 4))
    batch_mbboxes = np.zeros((max_bbox_per_scale, 4))
    batch_lbboxes = np.zeros((max_bbox_per_scale, 4))

    batch_image[:, :, :] = image
    batch_label_sbbox[:, :, :, :] = label_sbbox
    batch_label_mbbox[:, :, :, :] = label_mbbox
    batch_label_lbbox[:, :, :, :] = label_lbbox
    batch_sbboxes[:, :] = sbboxes
    batch_mbboxes[:, :] = mbboxes
    batch_lbboxes[:, :] = lbboxes

    return batch_image, batch_label_sbbox, batch_label_mbbox,
    batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes


def image_preprocess(image, bboxes, input_size):
    """
    进行数据增强，以及图片的resize, 包含对box的处理
    水平翻转，剪裁，随机迁移
    letterbox 图片输入前，尺寸变换
    input:
        image bboxes, input_size
    return:
        image bbox
    """
    image, bboxes = img.random_horizontal_flip(np.copy(image), np.copy(bboxes))
    image, bboxes = img.random_crop(np.copy(image), np.copy(bboxes))
    image, bboxes = img.random_translate(np.copy(image), np.copy(bboxes))
    image, bboxes = img.letterBox(np.copy(image), [input_size, input_size],
                                  np.copy(bboxes))
    return image, bboxes


def bboxes_preprocess(bboxes, labels, output_sizes, num_classes):
    '''
    input:
        gt_bboxes, output_sizes, num_classes
    output:
        计算loss中使用的bbox
    '''
    label = [
        np.zeros((output_sizes[i], output_sizes[i], cfg.YOLO.ANCHOR_PER_SCALE,
                  5 + num_classes)) for i in range(3)
    ]
    bboxes_xywh = [
        np.zeros((cfg.YOLO.MAX_BBOX_PER_SCALE, 4)) for _ in range(3)
    ]
    bbox_count = np.zeros((3, ))

    for bbox_index in range(len(bboxes)):
        bbox_coor = bboxes[bbox_index]
        bbox_class_ind = labels[bbox_index]

        # smooth label
        onehot = np.zeros(num_classes, dtype=np.float)
        onehot[bbox_class_ind] = 1.0
        uniform_distribution = np.full(num_classes,
                                       1.0 / num_classes)
        deta = 0.01
        smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

        # bbox [xcernter, ycenter, width, height]
        bbox_xywh = np.concatenate(
            [(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
             bbox_coor[2:] - bbox_coor[:2]],
            axis=-1)

        bbox_xywh_scaled = 1.0 * bbox_xywh[
            np.newaxis, :] / strides[:, np.newaxis]
        # [xmin, ymin, width, height]
        # ==========>
        # [
        #  [xmin/8,  ymin/8,  width/8,  height/8]
        #  [xmin/16, ymin/16, width/16, height/16]
        #  [xmin/32, ymin/32, width/32, height/32]
        # ]

        iou = []
        exist_positive = False
        for i in range(3):  # different ratio 8, 16, 32
            anchors_xywh = np.zeros((self.anchor_per_scale, 4))
            anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(
                np.int32) + 0.5
            anchors_xywh[:, 2:4] = self.anchors[i]

            iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :],
                                      anchors_xywh)
            iou.append(iou_scale)
            iou_mask = iou_scale > 0.3

            if np.any(iou_mask):
                xind, yind = np.floor(bbox_xywh_scaled[i,
                                                       0:2]).astype(np.int32)

                label[i][yind, xind, iou_mask, :] = 0
                label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                label[i][yind, xind, iou_mask, 4:5] = 1.0
                label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                bbox_count[i] += 1

                exist_positive = True

        if not exist_positive:
            best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
            best_detect = int(best_anchor_ind / self.anchor_per_scale)
            best_anchor = int(best_anchor_ind % self.anchor_per_scale)
            xind, yind = np.floor(bbox_xywh_scaled[best_detect,
                                                   0:2]).astype(np.int32)

            label[best_detect][yind, xind, best_anchor, :] = 0
            label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
            label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
            label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

            bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
            bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
            bbox_count[best_detect] += 1
    label_sbbox, label_mbbox, label_lbbox = label
    sbboxes, mbboxes, lbboxes = bboxes_xywh
    return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes