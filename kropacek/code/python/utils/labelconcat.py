import os
import sys

import cv2


def concat_labels(labels_to_concat, resulting_label, labels_folder_path):
    print("running label concat for path: " + labels_folder_path)
    labels = sorted(os.listdir(labels_folder_path))
    for label_name in labels:
        print("processing: " + label_name)
        label_path = os.path.join(labels_folder_path, label_name)
        label_img = cv2.imread(label_path)
        for label_num in labels_to_concat:
            label_img[label_img == int(label_num)] = int(resulting_label)
        cv2.imwrite(label_path, label_img)

    print("done!")


# Used to merge multiple labels into one specified
# labelconcat.py 4,5 4 c:/projects/ateroskleroza-data/histology_segmentation/datasets/hist175_classes5_size512x512_noMargin_stainHEVG/labels
if __name__ == "__main__":
    labels_to_concat = sys.argv[1].split(",")
    resulting_label = sys.argv[2]
    labels_folder_path = sys.argv[3]
    concat_labels(labels_to_concat, resulting_label, labels_folder_path)
