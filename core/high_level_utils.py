import numpy as np


def read_image_box_from_text(text_path):
    """
    :param text_path
    :returns : {image_path:(bboxes, labels)}
                bboxes -> [N,4],(x1, y1, x2, y2)
                labels -> [N,]
    """
    data = {}
    with open(text_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            example_line = line.split(' ')

            image_path = example_line[0]
            image_boxes = example_line[1:]

            bboxes = np.zeros([len(image_boxes), 4], dtype=np.float32)
            labels = np.zeros([len(image_boxes), ],  dtype=np.int64)

            for i in range(len(image_boxes)):
                box_info = image_boxes[i].split(",")
                bboxes[i] = box_info[0:4]
                labels[i] = box_info[4]

            data[image_path] = bboxes, labels
        return data
