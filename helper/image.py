import numpy as np
import random
import cv2


def random_horizontal_flip(image, bboxes=None):
    if random.random() > 0.5:
        image = image[:, ::-1, :]
        _, w, _ = image.shape

        if bboxes is not None:
            xmin = bboxes[:, 0]
            xmax = bboxes[:, 2]
            bboxes[:, 0] = w - xmax
            bboxes[:, 2] = w - xmin

    if bboxes is not None:
        return image, bboxes
    else:
        return image


def random_crop(image, bboxes):
    """
    np.random.randint()  [low, high) or [0, low)
    """

    H, W, C = image.shape
    xmin = min(bboxes[:, 0])
    ymin = min(bboxes[:, 1])
    xmax = max(bboxes[:, 2])
    ymax = max(bboxes[:, 3])

    xLeftCrop = np.random.randint(xmin) if xmin > 0 else 0
    yUpCrop = np.random.randint(ymin) if ymin > 0 else 0
    xRightCrop = np.random.randint(xmax + 1, W)
    yDownCrop = np.random.randint(ymax + 1, H)

    image = image[yUpCrop:yDownCrop, xLeftCrop:xRightCrop, :]
    bboxes = bboxes[:, 0] - xmin
    bboxes = bboxes[:, 1] - ymin
    bboxes = bboxes[:, 2] - xmin
    bboxes = bboxes[:, 3] - ymin
    return image, bboxes


def random_translate(image, bboxes):
    '''
    采用affine仿射变换，不影响bboxes的状态中，只做平移
    '''
    if random.random() < 0.5:
        H, W, _ = image.shape
        boundary = np.concatenate(
            [np.min(bboxes[:, 0:2], axis=0),
             np.max(bboxes[:, 2:4], axis=0)],
            axis=-1)

        tx = random.uniform(-(boundary[0] - 1), W - (boundary[2] + 1))
        ty = random.uniform(-(boundary[1] - 1), H - (boundary[3] + 1))

        M = np.array([1, 0, tx], [0, 1, ty])
        image = cv2.warpAffine(image, M, (W, H))

        bboxes = bboxes[:, [0, 2]] + tx
        bboxes = bboxes[:, [1, 3]] + ty

    return image, bboxes


def letterBox(image, target_size, gt_bboxes=None):
    """
    保持图片比例，缩放到target_size*target_size, 空余位置补0
    """
    h, w, _ = image.shape
    th, tw = target_size
    scale = min(th / h, tw / w)
    nh, nw = int(h * scale), int(w * scale)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (nh, nw))

    image_padded = np.full((th, tw, 3), 128)
    bh, bw = (th - nh) // 2, (tw - nw) // 2
    image_padded[bh:bh + nh, bw:bw + nw] = image
    image_padded = image_padded / 255.0

    if gt_bboxes is None:
        return image_padded
    else:
        gt_bboxes = gt_bboxes[:, [0, 2]] * scale + bw
        gt_bboxes = gt_bboxes[:, [1, 3]] * scale + bh
        return image_padded, gt_bboxes
