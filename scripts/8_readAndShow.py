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

Path = "/home/guo/Documents/Dataset/object-dataset"
imagefiles = glob.glob(os.path.join(Path, "*.jpg"))
labelfile = os.path.join(Path, "labels.csv")

imagelist = []
for file in imagefiles:
    imagelist.append(file.split("\n")[0].split("/")[-1])

imagebox = {}
with open(labelfile, "r+") as f:
    for line in f:
        item = line.split("\n")[0].split(" ")
        if not item[0] in imagebox:
            imagebox[item[0]] = []
        imagebox[item[0]].append(item[1:])

imagelist.sort()
imagefiles.sort()

for index in range(len(imagelist)):
    if imagelist[index] in imagebox:
        image = cv2.imread(imagefiles[index], 1)
        for box in imagebox[imagelist[index]]:
            xMin = int(box[0])
            yMin = int(box[1])
            xMax = int(box[2])
            yMax = int(box[3])

            cv2.rectangle(image, (xMin, yMin), (xMax, yMax), (0, 0, 255), 2)
            cv2.putText(image, box[5],
                        (xMin, yMin), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
        cv2.imwrite(Path+"/draw_"+imagelist[index], image)
        # cv2.namedWindow("image",cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("image", image)
        # print(imagelist[index])
        # key = cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # if key == "q":
        #     break

