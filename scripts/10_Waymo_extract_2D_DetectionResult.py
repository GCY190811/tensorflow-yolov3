import sys
sys.path.append("/root/gCode/waymo-open-dataset")

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import range_image_utils
import glob
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import itertools
import numpy as np
import tensorflow as tf
import cv2
import math
import os


def show_camera_image(camera_image, camera_labels, index, segmentName, cmap=None):
    """Show a camera image and the given camera labels."""

    img = tf.image.decode_jpeg(camera_image.image).numpy()

    labelfolder = "/opt/home/moDisk/Dataset/Waymo/ImgLabel/"+segmentName
    if not os.path.exists(labelfolder):
        os.makedirs(labelfolder)
    labelfile = labelfolder+"/"+segmentName.split("_", 2)[-1]+"_"+open_dataset.CameraName.Name.Name(
        camera_image.name)+"_"+str(index)+".txt"
    with open(labelfile, "w+") as wf:

        # Draw the camera labels.
        for camera_label in camera_labels:
            # Ignore camera labels that do not correspond to this camera. # unkown, vehicle, pedestrain, sign, cyclist
            if camera_label.name != camera_image.name:
                continue

            # Iterate over the individual labels.
            for label in camera_label.labels:
                # Draw the object bounding box.
                leftx = int(label.box.center_x - 0.5 * label.box.length)
                leftTop = int(label.box.center_y - 0.5 * label.box.width)
                cv2.rectangle(img, (leftx, leftTop), (leftx+int(label.box.length),
                                                      leftTop+int(label.box.width)), (0, 0, 255), 2)
                cv2.putText(img, "label: "+str(label.type), (leftx+5,
                                                             leftTop-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 128))
                line = str(label.type) + " " + str(label.box.center_x) + " " + str(
                    label.box.center_y) + " " + str(label.box.length) + " " + str(label.box.width) + "\n"
                wf.writelines(line)
    
    drawfolder = "/opt/home/moDisk/Dataset/Waymo/ImgDraw/"+segmentName
    if not os.path.exists(drawfolder):
        os.makedirs(drawfolder)
    cv2.imwrite(drawfolder+"/"+segmentName.split("_", 1)[1]+"_"+open_dataset.CameraName.Name.Name(
        camera_image.name)+"_draw_"+str(index)+".jpeg", img[:,:,[2,1,0]])

    Imgfolder = "/opt/home/moDisk/Dataset/Waymo/Img/"+segmentName
    if not os.path.exists(Imgfolder):
        os.makedirs(Imgfolder)
    cv2.imwrite(Imgfolder+"/"+segmentName.split("_", 1)[1]+"_"+open_dataset.CameraName.Name.Name(
        camera_image.name)+"_"+str(index)+".jpeg", tf.image.decode_jpeg(camera_image.image).numpy()[:,:,[2,1,0]])


def main():
    # tf.enable_eager_execution()
    tf.compat.v1.enable_eager_execution()

    folderPath = "/opt/home/moDisk/Dataset/Waymo/Waymo/training_0000_0005/training_0001"
    files = glob.glob(os.path.join(folderPath, "*.tfrecord"))
    print("parse tfrecord num: {}".format(len(files)))

    recordFileNum = 0

    for FILENAME in files:
        dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
        imagenum = 0
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            # (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

            print("Frame Context Name:")
            print(frame.context.name)
            segmentName = frame.context.name  # 视频段编号
            segmentName = folderPath.split("/")[-1]+"/"+segmentName
            print("segmentName:")
            print(segmentName)

            for index, image in enumerate(frame.images):  # 每一帧数据中 不同相机的图片
                print("Image num:{}".format(index))
                # input all this frame all labels
                show_camera_image(image, frame.camera_labels,
                                  imagenum, segmentName)
            imagenum += 1

            # if imagenum > 10:
            #     break

        print("New tfrecord!!!")
        recordFileNum += 1
        # if recordFileNum >= 2:
        #     break


if __name__ == "__main__":
    main()
