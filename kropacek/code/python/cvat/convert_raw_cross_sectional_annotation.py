import os
import cv2
import numpy as np
from pathlib import Path


from utils.cvat_utils import get_categories_to_colors
from utils.image_utils import segment2mask


def main():
    raw_annotations_path = Path(
        'c:/projects/ateroskleroza-data/in_vivo_us_segmentace/cross_sectional/raw_img_annotation')
    output_path = Path('c:/projects/ateroskleroza-data/in_vivo_us_segmentace/cross_sectional/converted_img_annotation')

    raw_annotations_file_names = os.listdir(raw_annotations_path)
    for raw_annotation_file_name in raw_annotations_file_names:
        raw_annotation_path = raw_annotations_path / raw_annotation_file_name
        output_annotation_path = output_path / raw_annotation_file_name
        raw_annotation_path_str = str(raw_annotation_path)
        print('processing {}'.format(raw_annotation_path_str))
        process_raw_annotation(output_annotation_path, cv2.imread(raw_annotation_path_str))


def get_dilatation_kernel():
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def process_raw_annotation(output_annotation_path, raw_annotation_img):
    category_to_colors = get_categories_to_colors()
    mask = segment2mask(raw_annotation_img, category_to_colors)

    vessel_mask = mask[:, :, 1].copy()
    lumen_mask = mask[:, :, 2].copy()
    plaque_mask = mask[:, :, 3].copy()

    structuring_element = get_dilatation_kernel()

    plaque_mask = cv2.dilate(plaque_mask, structuring_element)

    extracted_plaque_mask = plaque_mask + lumen_mask
    cv2.floodFill(extracted_plaque_mask, None, (0, 0), 1)

    plaque_mask += extracted_plaque_mask
    plaque_mask[plaque_mask == 0] = 2
    plaque_mask[plaque_mask == 1] = 0
    plaque_mask[plaque_mask > 1] = 1

    result = np.zeros(raw_annotation_img.shape)
    result[vessel_mask == 1] = category_to_colors['Vessel']
    result[lumen_mask == 1] = category_to_colors['Lumen']
    result[plaque_mask == 1] = category_to_colors['Plaque']

    cv2.imwrite(str(output_annotation_path), result)


if __name__ == '__main__':
    main()
    exit(0)
