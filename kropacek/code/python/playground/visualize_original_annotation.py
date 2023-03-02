from pathlib import Path

import cv2

from utils.dataset_utils import load_colors, load_text
from utils.visualization_utils import visualize_img_label


def main():
    img_path = Path('f:/ondra/ateroskleroza-data/histology_segmentation/data/svetla_tmava/svetla/data/10251_14_x226')
    metadata_path = Path('f:/ondra/ateroskleroza-data/histology_segmentation/datasets/playground')
    colors = load_colors(metadata_path / 'colors.txt')
    classes = load_text(metadata_path / 'classes.txt')

    image_name = '10251_14_x226.png'
    visualize_img_label(colors, classes,
                        cv2.imread(str(img_path / '10251_14_x226_original.png')),
                        cv2.imread(str(img_path / 'masks/all_masks_colored.png')),
                        image_name,
                        title='Original annotation: ')


if __name__ == '__main__':
    main()
    exit(0)
