import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math
import numpy as np
import cv2

leftX = 0
leftTop = 0
rightX = 0
rightBottom = 0


class object_bbox_udacity2:
    img = ''
    xMin = 0
    yMin = 0
    xMax = 0
    yMax = 0
    occluded = -1
    label = ''
    attribute = -1
    tXMin = 0
    tYMin = 0
    tXMax = 0
    tYMax = 0

    def __init__(self, name, xmin, ymin, xmax, ymax, label, tXMin, tYMin, tXMax, tYMax, occluded=-1, attribute=-1):
        self.img = name
        self.xMin = xmin
        self.yMin = ymin
        self.xMax = xmax
        self.yMax = ymax
        self.label = label
        self.occluded = occluded
        self.attribute = attribute
        self.tXMin = tXMin
        self.tYMin = tYMin
        self.tXMax = tXMax
        self.tYMax = tYMax

    def print(self):
        print("image: {}, xmin: {}, ymin: {}, xmax: {}, ymax: {}, label: {}, tXMin {}, tYMin {}, tXMax {}, tYMax {}".format(
              self.img, str(self.xMin), str(self.yMin), str(self.xMax), str(self.yMax), self.label, self.tXMin, self.tYMin, self.tXMax, self.tYMax))

    # def crop(self):


def define_args():
    parser = argparse.ArgumentParser(
        description='Crop Udacity Dataset2 to 1280*720, RGB2Gray, TF-yolo3 format')
    parser.add_argument(
        '-IIf', default='/home/guo/moDisk/Dataset/object-dataset', help='Input Image folder')
    parser.add_argument(
        '-IL', default="/home/guo/moDisk/Dataset/object-dataset/labels.csv", help='Input Label file')
    parser.add_argument(
        '-SCL', default="/home/guo/moDisk/Dataset/self_driving_car_dataset2_label.csv", help='Selected Class file')
    parser.add_argument(
        '-Of', default='/home/guo/moDisk/Dataset/object-dataset-crop',  help='Output folder')
    parser.add_argument('-Ch', default=720, type=int, help='Crop Height')
    parser.add_argument('-Cw', default=1280, type=int, help='Crop Width')
    return parser


def CropImg(ImgLists):
    for imgfile in ImgLists:
        img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
        imgHeight = len(img)
        imgWidth = len(img[0])
        img = np.array(img)
        leftX = imgWidth//2-args.Cw//2
        leftTop = imgHeight//2-args.Ch//2
        rightX = imgWidth//2+args.Cw//2
        rightBottom = imgHeight//2 + args.Ch//2

        new_img = img[imgHeight//2-args.Ch//2:imgHeight//2 + args.Ch //
                      2, imgWidth//2-args.Cw//2:imgWidth//2+args.Cw//2]
        imgNewFile = os.path.join(args.Of, imgfile.split("/")[-1])
        cv2.imwrite(imgNewFile, new_img)

        print("Crop: leftX {}, leftTop {}, rightX {}, rightBottom {}".
              format(leftX, leftTop, rightX, rightBottom))
        break


def Readfile(file):
    fileList = []
    with open(file, "r+") as f:
        for line in f:
            fileList.append(line.split("\n")[0:-1])
    return fileList


def CropBox(Labels, LabelClass, ImageLists):
    ImgLabel = {}
    for label in Labels:
        item = label[0].split(" ")
        if item[6][1:-1] not in LabelClass:  # 排除不要的类别标签
            continue
        frame = item[0]
        print("Crop: leftX {}, leftTop {}, rightX {}, rightBottom {}".
              format(leftX, leftTop, rightX, rightBottom))
        box = object_bbox_udacity2(frame, int(item[1]), int(
            item[2]), int(item[3]), int(item[4]), item[6][1:-1], leftX, leftTop, rightX, rightBottom)
        if not frame in ImgLabel:
            ImgLabel[frame] = []
        ImgLabel[frame].append(box)

    with open(os.path.join(args.Of, "dataset.txt"), "w+") as f:
        for imgPath in ImageLists:
            imageName = imgPath.split("\n")[0].split("/")[-1]
            line = imageName
            if imageName in ImgLabel:
                annotations = ImgLabel[imageName]
                for i in range(len(annotations)):
                    cropLabel = annotations[i].print()
                    break

            break


def main():
    global args
    parser = define_args()
    args = parser.parse_args()

    ImageLists = glob.glob(os.path.join(args.IIf, "*.jpg"))
    ImageLists.sort()
    print("Img number: {}".format(len(ImageLists)))
    CropImg(ImageLists)

    LabelClass = np.array(Readfile(args.SCL))
    LabelClass = np.ndarray.flatten(LabelClass).tolist()
    print("Class: {}".format(LabelClass))
    Labels = Readfile(args.IL)

    CropBox(Labels, LabelClass, ImageLists)


if __name__ == "__main__":
    main()
