import os
import sys

import cv2


def resize(height, width, input_folder_path, output_folder_path):
    print("running resize for path: " + input_folder_path)
    labels = sorted(os.listdir(input_folder_path))
    for img_name in labels:
        print("processing: " + img_name)
        img_path = input_folder_path + '/' + img_name
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output_folder_path, img_name), resized)
    print("done!")


# resize.py 512 512 c:/projects/ateroskleroza-data/histology_segmentation/datasets/hist175_classes8_size512x512_noMargin_stainHEVG/labels c:/projects/ateroskleroza-data/histology_segmentation/datasets/hist175_classes8_size512x512_noMargin_stainHEVG/labels
if __name__ == "__main__":
    height = int(sys.argv[1])
    width = int(sys.argv[2])
    input_folder_path = sys.argv[3]
    output_folder_path = sys.argv[4]

    resize(height, width, input_folder_path, output_folder_path)
