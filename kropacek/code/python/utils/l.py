import unicodedata
import cv2
import numpy as np
import os
from PIL import Image
dataset_dir = 'histology_rotated'
original_dir = 'C:/bakalarka'
for i in os.listdir(original_dir + '/' + dataset_dir):
    for img_name in os.listdir(original_dir + '/' + dataset_dir + '/' + i):
        npMask = np.array(Image.open(original_dir + '/' + dataset_dir + '/' + i + '/' + img_name))
        npMask[npMask == 0] = 255
        orig_img = Image.fromarray(npMask)
        orig_img.save(original_dir + '/rotatedwhite/' + i + '/' + img_name)

