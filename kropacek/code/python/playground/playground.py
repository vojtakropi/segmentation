import os
from pathlib import Path

import cv2

from utils.dataset_utils import load_colors, load_text
from utils.visualization_utils import visualize_img_label


def main():
    img_path = Path('F:/ondra/ateroskleroza-data/histology_segmentation/data/patient_to_img_transformed')
    labels_path = Path(
        'f:/ondra/ateroskleroza-data/histology_segmentation/datasets/2021_04_05_hist55_pacient18_merged_prob_new/labels-visible')
    metadata_path = Path('F:/ondra/ateroskleroza-data/histology_segmentation/datasets/playground')
    output_dir = Path(
        'F:/ondra/ateroskleroza-data/histology_segmentation/visualisations/2021_04_05_hist55_pacient18_merged_prob_new')
    colors = load_colors(metadata_path / 'colors.txt')
    classes = load_text(metadata_path / 'classes.txt')

    original_file_names = os.listdir(img_path)
    label_file_names = os.listdir(labels_path)
    for label_file_name in label_file_names:
        label_path = labels_path / label_file_name
        he_image_name = None
        vg_image_name = None
        for original_file_name in original_file_names:
            if '_label' in original_file_name:
                continue
            if he_image_name is not None and vg_image_name is not None:
                break
            if label_file_name.replace('.png', '_HE') in original_file_name:
                he_image_name = original_file_name

            if label_file_name.replace('.png', '_VG') in original_file_name:
                vg_image_name = original_file_name

        if he_image_name is not None:
            image = cv2.imread(str(img_path / he_image_name))
            segmentation = cv2.imread(str(label_path))
            visualize_img_label(colors, classes,
                                image,
                                segmentation,
                                he_image_name,
                                output_path=output_dir,
                                title='')

        if vg_image_name is not None:
            image = cv2.imread(str(img_path / vg_image_name))
            segmentation = cv2.imread(str(label_path))
            visualize_img_label(colors, classes,
                                image,
                                segmentation,
                                vg_image_name,
                                output_path=output_dir,
                                title='')


if __name__ == '__main__':
    main()
    exit(0)
