import os
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import label
from matplotlib import pyplot as plt

from cvat.utils import get_categories_to_colors, segment2mask, points2mask, mask2points


def main():
    raw_annotations_path = Path(
        'c:/projects/ateroskleroza-data/in_vivo_us_segmentace/longitudinal/raw_img_annotation')
    output_path = Path('c:/projects/ateroskleroza-data/in_vivo_us_segmentace/longitudinal/converted_img_annotation')

    raw_annotations_file_names = os.listdir(raw_annotations_path)
    for raw_annotation_file_name in raw_annotations_file_names:
        raw_annotation_path = raw_annotations_path / raw_annotation_file_name
        output_annotation_path = output_path / raw_annotation_file_name
        raw_annotation_path_str = str(raw_annotation_path)
        print('processing {}'.format(raw_annotation_path_str))
        process_raw_annotation(output_annotation_path, cv2.imread(raw_annotation_path_str))


def get_dilatation_kernel():
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def show_img(img):
    plt.imshow(img)
    plt.show()


def process_raw_annotation(output_annotation_path, raw_annotation_img):
    category_to_colors = get_categories_to_colors()
    img_shape = raw_annotation_img.shape[:2]
    mask = segment2mask(raw_annotation_img, category_to_colors)

    vessel_mask = mask[:, :, 1].copy()
    lumen_mask = mask[:, :, 2].copy()
    plaque_mask = mask[:, :, 3].copy()

    lumen_contour = get_contour_from_mask(lumen_mask)
    lumen_contour_points = mask2points(lumen_contour)

    lumen_middle_contour_mask = get_middle_contour_mask(lumen_contour_points, lumen_mask)

    lumen_middle_contour_points = mask2points(lumen_middle_contour_mask)

    is_near_lumen = is_above_middle_lumen(lumen_contour_points, lumen_middle_contour_points)

    near_lumen_contour_points = lumen_contour_points[:, is_near_lumen].copy()
    near_lumen_contour_mask = points2mask(near_lumen_contour_points, img_shape)
    near_lumen_contour_mask = get_largest_contour(near_lumen_contour_mask)
    near_lumen_contour_points = mask2points(near_lumen_contour_mask)

    far_lumen_contour_points = lumen_contour_points[:, ~is_near_lumen].copy()
    far_lumen_contour_mask = points2mask(far_lumen_contour_points, img_shape)

    far_lumen_contour_mask = get_largest_contour(far_lumen_contour_mask)
    far_lumen_contour_points = mask2points(far_lumen_contour_mask)

    plaque_contour_points = mask2points(plaque_mask)

    is_near_plaque = is_above_middle_lumen(plaque_contour_points, lumen_middle_contour_points)

    near_plaque_contour_points = plaque_contour_points[:, is_near_plaque].copy()

    if near_plaque_contour_points.size != 0:
        filling_points = get_filling_points(near_plaque_contour_points, near_lumen_contour_points)
        near_plaque_contour_points = np.hstack((near_plaque_contour_points, filling_points))
        near_plaque_contour_mask = points2mask(near_plaque_contour_points, img_shape)
        near_plaque_mask = ndimage.binary_fill_holes(
            near_plaque_contour_mask + near_lumen_contour_mask).astype('uint8')
        near_plaque_mask = near_plaque_mask - near_plaque_contour_mask - near_lumen_contour_mask
        near_plaque_mask = cv2.dilate(near_plaque_mask, get_dilatation_kernel()).astype('uint8')
    else:
        near_plaque_mask = np.zeros(img_shape)

    far_plaque_contour_points = plaque_contour_points[:, ~is_near_plaque].copy()

    if far_plaque_contour_points.size != 0:
        filling_points = get_filling_points(far_lumen_contour_points, far_plaque_contour_points)
        far_plaque_contour_points = np.hstack((far_plaque_contour_points, filling_points))
        far_plaque_contour_mask = points2mask(far_plaque_contour_points, img_shape)
        far_plaque_mask = ndimage.binary_fill_holes(far_plaque_contour_mask + far_lumen_contour_mask).astype('uint8')
        far_plaque_mask = far_plaque_mask - far_plaque_contour_mask - far_lumen_contour_mask
        far_plaque_mask = cv2.dilate(far_plaque_mask, get_dilatation_kernel()).astype('uint8')
    else:
        far_plaque_mask = np.zeros(img_shape)

    plaque_mask = near_plaque_mask + far_plaque_mask

    vessel_mark_contour_label = label(vessel_mask)
    near_vessel_mark_contour_mask = (vessel_mark_contour_label == 1).astype('int')
    near_vessel_mark_contour_points = mask2points(near_vessel_mark_contour_mask)
    far_vessel_mark_contour_mask = (vessel_mark_contour_label == 2).astype('int')
    far_vessel_mark_contour_points = mask2points(far_vessel_mark_contour_mask)

    if len(np.unique(near_vessel_mark_contour_mask)) > 1 and len(np.unique(far_vessel_mark_contour_mask)) > 1:
        near_widths = contour_widths(near_vessel_mark_contour_points, near_lumen_contour_points)
        far_widths = contour_widths(far_vessel_mark_contour_points, far_lumen_contour_points)

        widths = np.hstack((
            near_widths,
            far_widths
        ))

        median_width = int(np.floor(np.median(widths)))

        near_vessel_contour_points = near_lumen_contour_points.copy()

        near_vessel_contour_points[0] -= median_width

        near_vessel_contour_mask = points2mask(near_vessel_contour_points, img_shape)
        near_vessel_mask = ndimage.binary_fill_holes(
            np.sign(near_vessel_contour_mask + near_lumen_contour_mask)).astype('uint8')
        near_vessel_mask = near_vessel_mask - np.sign(near_vessel_contour_mask + near_lumen_contour_mask)

        near_vessel_mask = cv2.dilate(near_vessel_mask, get_dilatation_kernel()).astype('uint8')

        far_vessel_contour_points = far_lumen_contour_points.copy()

        far_vessel_contour_points[0] += median_width
        far_vessel_contour_mask = points2mask(far_vessel_contour_points, img_shape)

        far_vessel_mask = ndimage.binary_fill_holes(
            np.sign(far_vessel_contour_mask + far_lumen_contour_mask)).astype('uint8')

        far_vessel_mask = far_vessel_mask - np.sign(far_vessel_contour_mask + far_lumen_contour_mask)

        far_vessel_mask = cv2.dilate(far_vessel_mask, get_dilatation_kernel())

        vessel_mask = near_vessel_mask + far_vessel_mask
        vessel_mask -= np.logical_and(lumen_mask, vessel_mask).astype('uint8')
        vessel_mask -= np.logical_and(plaque_mask, vessel_mask).astype('uint8')
    else:
        vessel_mask = np.zeros(img_shape)

    lumen_mask -= np.logical_and(lumen_mask, plaque_mask).astype('uint8')

    result = np.zeros(raw_annotation_img.shape, dtype=np.uint8)
    result[vessel_mask == 1] = category_to_colors['Vessel']
    result[plaque_mask == 1] = category_to_colors['Plaque']
    result[lumen_mask == 1] = category_to_colors['Lumen']

    cv2.imwrite(str(output_annotation_path), result)


def width_in_column(contour_points_0, contour_points_1, y):
    points_0 = points_in_column(contour_points_0, y)
    points_1 = points_in_column(contour_points_1, y)

    width = 0

    if np.max(points_0[0]) < np.min(points_1[0]):
        width = np.min(points_1[0]) - np.max(points_0[0]) - 1

    if np.max(points_1[0]) < np.min(points_0[0]):
        width = np.min(points_0[0]) - np.max(points_1[0]) - 1

    return width


def contour_widths(contour_points_0, contour_points_1):
    y_left = max(
        left_points(contour_points_0)[1, 0],
        left_points(contour_points_1)[1, 0]
    )

    y_right = min(
        right_points(contour_points_0)[1, 0],
        right_points(contour_points_1)[1, 0]
    )

    widths = np.zeros(y_right - y_left + 1, dtype='int')

    i = 0
    for y in range(y_left, y_right + 1):
        widths[i] = width_in_column(contour_points_0, contour_points_1, y)
        i += 1

    return widths


def get_center_point(contour_points):
    return np.round(np.mean(contour_points, axis=1)).astype('int')


def get_contour_from_mask(mask):
    result = np.zeros(mask.shape, dtype=np.uint8)
    cnt, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(result, cnt, 0, 1, 1)
    return result


def get_largest_contour(mask):
    result = np.zeros(mask.shape, dtype=np.uint8)
    cnt, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_cnt = max(cnt, key=cv2.contourArea)
    cv2.drawContours(result, [largest_cnt], 0, 1, 1)
    return result


def get_middle_contour_mask(contour_points, mask):
    y_left = np.min(contour_points[1])
    y_right = np.max(contour_points[1])

    middle_contour_mask = np.zeros(mask.shape)

    for y in range(y_left, y_right + 1):
        column = mask[:, y]

        x_middle = int(np.round(np.mean(np.where(column))))

        middle_contour_mask[x_middle, y] = 1

    return middle_contour_mask


def points_in_column(points, y):
    indexes = np.where(points[1] == y)[0]

    return np.array([(points[0, i], points[1, i]) for i in indexes]).T


def is_above_middle_lumen(contour_points, lumen_middle_contour_points):
    n_points = contour_points.shape[1]

    is_above_middle = np.zeros(n_points, dtype='bool')

    for i in range(n_points):
        x = contour_points[0, i]
        y = contour_points[1, i]

        x_middle = points_in_column(lumen_middle_contour_points, y)[0]

        is_above_middle[i] = (x < x_middle)

    return is_above_middle


def filling_points_in_column(upper_contour_points, lower_contour_points, y):
    if not is_first_contour_upper(upper_contour_points, lower_contour_points):
        # swap pointers
        upper_contour_points, lower_contour_points = lower_contour_points, upper_contour_points

    upper_points = points_in_column(upper_contour_points, y)
    lower_points = points_in_column(lower_contour_points, y)

    x_max_upper = np.max(upper_points[0])
    x_min_lower = np.min(lower_points[0])

    filling_points = np.arange(x_max_upper + 1, x_min_lower)

    filling_points = np.vstack((
        filling_points,
        y * np.ones(filling_points.shape, dtype='int')
    ))

    return filling_points


def get_filling_points(upper_contour_points, lower_contour_points):
    y_left = np.max((
        left_points(upper_contour_points)[1, 0],
        left_points(lower_contour_points)[1, 0],
    ))

    left_filling_points = filling_points_in_column(upper_contour_points, lower_contour_points, y_left)

    y_right = np.min((
        right_points(upper_contour_points)[1, 0],
        right_points(lower_contour_points)[1, 0],
    ))

    right_filling_points = filling_points_in_column(upper_contour_points, lower_contour_points, y_right)

    return np.hstack((left_filling_points, right_filling_points))


def get_center_point(contour_points):
    return np.round(np.mean(contour_points, axis=1)).astype('int')


def is_first_contour_upper(contour_points_0, contour_points_1):
    center_point_0 = get_center_point(contour_points_0)
    center_point_1 = get_center_point(contour_points_1)

    return center_point_0[0] < center_point_1[0]


def left_points(contour_points):
    y = np.min(contour_points[1])
    return points_in_column(contour_points, y)


def right_points(contour_points):
    y = np.max(contour_points[1])
    return points_in_column(contour_points, y)


if __name__ == '__main__':
    main()
    exit(0)
