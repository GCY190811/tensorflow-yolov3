import os
import numpy as np
import argparse
import tensorflow as tf

"""
read binary mnist file
save tfrecord file
"""


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mnistFolder", help='mnist folder')
    args = parser.parse_args()
    return args


def convert(imgf, labelf, n, outf=None):
    f = open(imgf, "rb")

    l = open(labelf, "rb")

    f.read(16)  # 取出字节数
    l.read(8)
    images = []

    for i in range(n):
        # ord('a') -> 97  ord('\x0a') -> 10; 　　相反：char(97) -> 'a'
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    if outf:
        o = open(outf, "w")
        for image in images:
            o.write(",".join(str(pix) for pix in image)+"\n")
        o.close()

    f.close()
    l.close()
    return images


def saveRecord(images, labels, num_tfrecords, pathPrefix):
    images_num = images.shape[0]
    print("==>> Processing %d images" % images_num)
    per_tfrecord_images = images_num // num_tfrecords

    n = 0
    # 最后一个tfrecord文件包含少量图片
    while n <= num_tfrecords:
        tfrecord_file = pathPrefix+"%04d.tfrecords" % n
        with tf.python_io.TFRecordWriter(tfrecord_file) as record_writer:
            st = n*per_tfrecord_images
            en = (
                n+1)*per_tfrecord_images if n < num_tfrecords else images_num
            for i in range(st, en):
                image = images[i].flatten()
                label = labels[i]

                # tf.train.Features(feature={...})
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        # 其他数据集图片可以不用读出,直接保存为二进制类型格式
                        'image': tf.train.Feature(int64_list=tf.train.Int64List(value=image)),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    }
                ))

                record_writer.write(example.SerializeToString())
            print(">> Saving %5d images in %s" % (en-st, tfrecord_file))
            n += 1


def main():
    args = parser_args()

    mnistTrainImages = os.path.join(
        args.mnistFolder, "train-images.idx3-ubyte")
    mnistTrainLabels = os.path.join(
        args.mnistFolder, "train-labels.idx1-ubyte")
    mnistTestImages = os.path.join(args.mnistFolder, "t10k-images.idx3-ubyte")
    mnistTestLabels = os.path.join(args.mnistFolder, "t10k-labels.idx1-ubyte")

    test_info = convert(mnistTestImages, mnistTestLabels, 10000)
    testImage = np.array(test_info)[:, 1:]
    testLabel = np.array(test_info)[:, 0:1]
    saveRecord(testImage, testLabel, 10, os.path.join(
        args.mnistFolder, "test/mnistTest_"))

    train_info = convert(mnistTrainImages, mnistTrainLabels, 60000)
    trainImage = np.array(train_info)[:, 1:]
    trainLabel = np.array(train_info)[:, 0:1]
    saveRecord(trainImage, trainLabel, 20,
               os.path.join(args.mnistFolder, "train/mnistTrain_"))


if __name__ == "__main__":
    main()
