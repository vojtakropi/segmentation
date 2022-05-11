import os
import cv2.cv2 as cv2
import numpy as np
from lxml import etree
from shapely import geometry
from cvat.mapper.cvat_mapper import map_to_xml
from cvat.model.cvat import Label, Image, Polygon, Annotations
from utils.image_utils import mask_by_color


def convert_mask_to_cvat_xml(id, img_name, input_img):
    label_list = get_label_list()
    background_label = get_background_label()
    base_label = get_base_layer_label()

    height, width, dim = input_img.shape
    all_holes = get_holes_rgb(input_img, background_label.label_img_color)

    background = mask_by_color(input_img, background_label.label_img_color)
    background[all_holes == 255] = 0
    base_layer = cv2.bitwise_not(background)

    cvat_img = Image(id, img_name, width, height)

    polygons = list()
    for label in label_list:
        if label == background_label or (base_label and label == base_label):
            continue
        mask = mask_by_color(input_img, label.label_img_color)
        holes = get_holes_bin(mask, 0)

        while np.count_nonzero(holes) > 0:
            #if label.label_img_color == (0, 255, 0):
            #    break
           # else:
            mask[holes == 255] = 255
            holes = get_holes_bin(mask, 0)
        polygons.extend(get_polygons(mask, label, 0))
    if background_label:
        polygons.extend(get_polygons(all_holes, background_label, 0))
    if base_label:
        polygons.extend(get_polygons(base_layer, base_label, 0))
    z_order = 0
    polygons.sort(key=lambda item: len(item.points), reverse=True)
    for polygon in polygons:
        polygon.z_order = z_order
        cvat_img.add_polygon(polygon)
        z_order += 1
    return cvat_img


def get_polygons(mask, label: Label, z_order: int):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = list()
    for cnt in contours:
        cnt
        cnt = np.squeeze(cnt)
        # vic nez 2 body aby mohl byt vyroben polygon
        if cnt.ndim != 2 or cnt.shape[0] < 3:
            continue
        poly = geometry.Polygon(cnt)
        poly = poly.simplify(1.0, preserve_topology=True)
        if poly.is_empty is False:
            if poly.geom_type == 'MultiPolygon':
                for sub_poly in list(poly):
                    polygons.append(Polygon(label.name, list(sub_poly.exterior.coords), z_order))
            else:
                polygons.append(Polygon(label.name, list(poly.exterior.coords), z_order))
    return polygons


def get_holes_rgb(input_img, background_color):
    background = mask_by_color(input_img, background_color)
    background = cv2.bitwise_not(background)
    cv2.floodFill(background, None, (0, 0), 255)
    return cv2.bitwise_not(background)


def get_holes_bin(input_img, background_value):
    background = np.zeros(input_img.shape, dtype=np.uint8)
    background[input_img == background_value] = 255
    background = cv2.bitwise_not(background)
    cv2.floodFill(background, None, (0, 0), 255)
    return cv2.bitwise_not(background)


def get_background_label():
    return Label('Holes', '#000000', (0, 0, 0))


def get_base_layer_label():
    return Label('Plát', '#ffc000', (185, 255, 30))


def get_label_list():
    return list([get_base_layer_label(),
                 Label('Ateromová tkáň', '#33ddff', (0, 0, 255)),
                 Label('Kalcifikace', '#3d3df5', (255, 0, 0)),
                 #   Label('Trombóza - stará', '#b25050', (127, 0, 255)),
                 # Label('Makrofágy', '#a58e8e', (0, 165, 255)),
                 # Label('Hemosiderin siderofágy', '#e7dddd', (0, 255, 255)),
                 # Label('Lumen - upraven', '#285c28', (128, 128, 128)),
                 # Label('Novotvorba cév', '#584e4e', (0, 85, 0)),
                 Label('Lumen', '#dfdf72', (255, 0, 255)),
                 # Label('Krvácení - staré', '#eea7a7', (0, 0, 85)),
                 #  Label('Krvácení - čerstvé', '#fc0029', (0, 170, 0)),
                 #  Label('Zánět', '#ff00fa', (0, 85, 255)),
                 # Label('Trombóza - čerstvá', '#ff7700', (255, 255, 0)),
                 Label('Fibrózní vazivo', '#aaf0d1', (0, 255, 0)),
                 get_background_label()
                 ])


def main():
    dataset_dir = 'test_images'
    original_dir = 'C:/bakalarka/he/datasets/HE'
    labels_dir = 'test_labels'
    imgs = list()

    for img_name in os.listdir(original_dir + '/' + labels_dir):
        path_to_img = (original_dir + '/' + dataset_dir + '/' + img_name)
        path_to_img_label = (original_dir + '/' + labels_dir + '/' + img_name)
        label = cv2.imread(path_to_img_label)
        orig_img = cv2.imread(path_to_img)
        orig_shape = orig_img.shape
        label = cv2.resize(label, (orig_shape[0], orig_shape[1]), interpolation=cv2.INTER_NEAREST)
        img = convert_mask_to_cvat_xml(0, img_name, label)
        imgs.append(img)

    annotations = Annotations(None, imgs)
    xml = map_to_xml(annotations)
    with open("C:/bakalarka/segementation_output_he.xml", "w", encoding='utf-8') as f:
        f.write(etree.tostring(xml, pretty_print=True).decode())
    #    tree = ElementTree(xml)
    #    tree.write(f, encoding='utf-8')


if __name__ == '__main__':
    main()
    exit(0)
