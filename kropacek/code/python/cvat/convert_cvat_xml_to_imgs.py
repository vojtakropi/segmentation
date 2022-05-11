from pathlib import Path

import cv2
import numpy as np
from lxml import etree

from utils.cvat_utils import get_categories_to_colors


def main():
    cvat_xml_path_str = 'c:/projects/ateroskleroza-data/in_vivo_us_segmentace/longitudinal/raw_xml_annotation/annotations_kozel.xml'
    output_path = Path('c:/projects/ateroskleroza-data/in_vivo_us_segmentace/longitudinal/raw_img_annotation')

    annotations = process_cvat_xml(cvat_xml_path_str)
    for annotation in annotations:
        mask = create_mask(annotation)
        file_name = annotation['name']
        save_mask(output_path, file_name, mask)
        print('processed {}'.format(file_name))


def save_mask(path, filename, mask):
    full_path = path / filename
    cv2.imwrite(str(full_path), mask)


def create_mask(annotation):
    height = annotation['height']
    width = annotation['width']
    mask = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    thickness = 1
    categories_to_colors = get_categories_to_colors()
    for shape in annotation['shapes']:
        points = [tuple(map(float, p.split(','))) for p in shape['points'].split(';')]
        points = np.array([(int(p[0]), int(p[1])) for p in points])
        points = points.astype(int)
        color = categories_to_colors[shape['label']]
        if shape['type'] is 'polyline':
            mask = cv2.polylines(mask, [points], False, color=color, thickness=thickness)
        else:
            mask = cv2.drawContours(mask, [points], -1, color=color, thickness=thickness)
            mask = cv2.fillPoly(mask, [points], color=color)
    return mask


def process_cvat_xml(cvat_xml_path_str):
    root = etree.parse(cvat_xml_path_str).getroot()
    anno = []

    image_tag_xpath = './/image'
    for image_tag in root.iterfind(image_tag_xpath):
        image = {}
        for key, value in image_tag.items():
            image[key] = value
        image['shapes'] = []
        for poly_tag in image_tag.iter('polygon'):
            polygon = {'type': 'polygon'}
            for key, value in poly_tag.items():
                polygon[key] = value
            image['shapes'].append(polygon)
        for polyline_tag in image_tag.iter('polyline'):
            polyline = {'type': 'polyline'}
            for key, value in polyline_tag.items():
                polyline[key] = value
            image['shapes'].append(polyline)
        for box_tag in image_tag.iter('box'):
            box = {'type': 'box'}
            for key, value in box_tag.items():
                box[key] = value
            box['points'] = "{0},{1};{2},{1};{2},{3};{0},{3}".format(
                box['xtl'], box['ytl'], box['xbr'], box['ybr'])
            image['shapes'].append(box)
        image['shapes'].sort(key=lambda x: int(x.get('z_order', 0)))
        anno.append(image)
    return anno


if __name__ == '__main__':
    main()
    exit(0)
