import os

import cv2.cv2 as cv2
import numpy as np
from tifffile import tifffile

from utils.image_utils import get_visible_label_max


def transform_imgs():
    img_names = os.listdir(input_path)
    for img_name in img_names:
        img_path = input_path + '/' + img_name
        label = tifffile.imread(str(img_path)) / 255
        label = label.astype(np.bool)
        result = get_visible_label_max(label)
        img_name = img_name.replace('.tiff', '.png')
        cv2.imwrite(output_path + '/' + img_name, result)


if __name__ == "__main__":
    dataset_path = 'C:/bakalarka/he/datasets/HE'
    input_path = dataset_path + '/images'
    output_path = dataset_path + '/labels'
    transform_imgs()

    exit(0)
