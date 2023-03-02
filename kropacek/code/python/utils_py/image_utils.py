import numpy as np
from skimage import morphology


def get_rgb_mask_colors():
    return ((get_background_color(), (50, 50, 50)),
            ((0, 0, 255), (203, 204, 255)),
            ((0, 255, 0), (144, 255, 144)),
            ((255, 0, 0), (255, 216, 173)),
            ((0, 0, 85), (0, 0, 120)),
            ((0, 170, 0), (0, 170, 70)),
            ((255, 0, 255), (200, 0, 200)),
            (get_plaque_color(), (185, 230, 90)),
            ((128, 128, 128), (160, 160, 160)))


def get_visible_label_max(maximum_per_class):
    result = np.zeros((maximum_per_class.shape[0], maximum_per_class.shape[1], 3), dtype=np.uint8)
    rgb_mask_colors = get_rgb_mask_colors()
    labels = list()
    for i in range(0, maximum_per_class.shape[2]):
        labels.append({'label': i, 'data': maximum_per_class[:, :, i]})
    labels = sorted(labels, key=lambda item: np.count_nonzero(item['data']), reverse=True)
    background_color = get_background_color()
    plaque_color = get_plaque_color()
    # bg index
    background_indices = next(filter(lambda item: item['label'] == 0, labels))['data']
    # plaque_idx
    # plaque_indices = next(filter(lambda item: item['label'] == 7, labels))['data']
    #
    # # plaque must go first
    # if np.count_nonzero(plaque_indices) > 0:
    #     result[plaque_indices] = plaque_color
    # order is then done by area of labels
    for label in labels:
        idx = label['label']
        color = rgb_mask_colors[idx][0]
        max_indices = maximum_per_class[:, :, idx]
        if color == background_color or color == plaque_color:
            continue
        result[max_indices] = rgb_mask_colors[idx][0]
    # holes last
    if np.count_nonzero(background_indices) > 0:
        result[background_indices] = background_color
    return result


def contours_morphology(
        mask,
        colors,
        radius=3,
        background=None
):
    structuring_element = morphology.disk(radius=radius)

    if background is None:
        contours = np.zeros(
            (mask.shape[0], mask.shape[1], 3), dtype='uint8')
    else:
        contours = background.copy()

    for i in range(mask.shape[2]):
        mask_erosion = morphology.erosion(
            mask[:, :, i], structuring_element)

        mask_erosion_diff = mask[:, :, i] - mask_erosion

        contours[mask_erosion_diff.astype('bool')] = colors[i]

    return contours


def unique_indexes(mask):
    idxs = []
    for i in range(mask.shape[2]):
        if mask[:, :, i].sum() != 0:
            idxs = idxs + [i]
    return idxs


def get_visible_label_pred(img, maximum_per_class):
    result = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
    rgb_mask_colors = get_rgb_mask_colors()
    for i in range(0, img.shape[2]):
        sub = img[:, :, i]
        idx_one = sub == 1.0
        idx_half = sub == 0.5
        result[idx_one] = rgb_mask_colors[i][0]
        result[idx_half] = rgb_mask_colors[i][1]
    return result


def segment2mask(segmentation, colors):
    mask = np.zeros(
        (segmentation.shape[0],
         segmentation.shape[1],
         len(colors)),
        dtype='uint8')

    for i, color in enumerate(colors):
        mask[:, :, i] = np.all(segmentation == color, axis=2)
    return mask


def mask_by_color(img, color):
    mask = np.zeros(img.shape[:-1], dtype=np.uint8)
    mask[np.all(img == color, axis=2)] = 255
    return mask


def mask2points(mask):
    points = np.array(np.where(mask == 1))
    return points


def points2mask(points, shape):
    mask = np.zeros(shape, dtype='uint8')
    for i in range(points.shape[1]):
        mask[points[0, i], points[1, i]] = 1
    return mask


def get_background_color():
    return 0, 0, 0


def get_plaque_color():
    return 185, 255, 30
