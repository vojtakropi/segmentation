import cv2

import numpy as np
import skimage
import random

from skimage import morphology

from utils_py.visualization_utils import tech_show_image


def get_target_size():
    return 512, 512


def main():
    print("generator started")
    vg_bg_img, he_bg_img = get_background_images()
    vg_plaque_img, vg_plaque_contour, he_plaque_img, he_plaque_contour, plaque_mask, plaque_contour_mask = generate_plaque_images()
    vg_fibrous_lig_img, he_fibrous_lig_img, fibrous_lig_mask = generate_fibrous_lig_images(plaque_mask)
    vg_atheroma_img, he_atheroma_img, atheroma_mask = generate_atheroma_images(plaque_mask)
    vg_lumen_img, he_lumen_img, lumen_mask = generate_lumen_images(vg_bg_img, he_bg_img, plaque_mask)

    calc_img, calc_mask = generate_calcification(plaque_mask)

    vg_img, vg_mask = combine(vg_bg_img, vg_plaque_img, plaque_mask, vg_fibrous_lig_img, fibrous_lig_mask,
                              vg_atheroma_img, atheroma_mask, vg_lumen_img, lumen_mask, vg_plaque_contour,
                              plaque_contour_mask)

    he_img, he_mask = combine_vg(he_bg_img, he_plaque_img, plaque_mask, he_fibrous_lig_img, fibrous_lig_mask,
                     he_atheroma_img, atheroma_mask, he_lumen_img, lumen_mask, he_plaque_contour,
                     plaque_contour_mask, calc_img, calc_mask)

    tech_show_image(smoothen(vg_img, 3))
    tech_show_image(vg_mask)
    tech_show_image(smoothen(he_img, 3))
    tech_show_image(he_mask)
    print("generator finished")


def combine_vg(bg_img, plaque_img, plaque_mask, fibrous_lig_img, fibrous_lig_mask, atheroma_img, atheroma_mask,
                  lumen_img, lumen_mask, plaque_contour, plaque_contour_mask, calc_img, calc_mask):
    bg_img[plaque_mask == 255] = plaque_img[plaque_mask == 255]
    bg_img[fibrous_lig_mask == 255] = fibrous_lig_img[fibrous_lig_mask == 255]
    bg_img[atheroma_mask == 255] = atheroma_img[atheroma_mask == 255]
    bg_img[calc_mask == 255] = calc_img[calc_mask == 255]
    bg_img[lumen_mask == 255] = lumen_img[lumen_mask == 255]
    bg_img[plaque_contour_mask == 255] = plaque_contour[plaque_contour_mask == 255]

    fibrous_lig_mask, atheroma_mask, lumen_mask, calc_mask = \
        apply_masks_deformation(fibrous_lig_mask, atheroma_mask, lumen_mask, calc_mask)

    mask = plaque_mask.copy()
    mask[mask == 255] = 1
    mask[(fibrous_lig_mask == 255) & (plaque_mask == 255)] = 2
    mask[(atheroma_mask == 255) & (plaque_mask == 255)] = 3
    mask[(calc_mask == 255) & (plaque_mask == 255)] = 4
    mask[(lumen_mask == 255) & (plaque_mask == 255)] = 5
    mask[plaque_contour_mask == 255] = 1

    return bg_img, mask


def combine(bg_img, plaque_img, plaque_mask, fibrous_lig_img, fibrous_lig_mask, atheroma_img, atheroma_mask,
                  lumen_img, lumen_mask, plaque_contour, plaque_contour_mask):
    bg_img[plaque_mask == 255] = plaque_img[plaque_mask == 255]
    bg_img[fibrous_lig_mask == 255] = fibrous_lig_img[fibrous_lig_mask == 255]
    bg_img[atheroma_mask == 255] = atheroma_img[atheroma_mask == 255]
    bg_img[lumen_mask == 255] = lumen_img[lumen_mask == 255]
    bg_img[plaque_contour_mask == 255] = plaque_contour[plaque_contour_mask == 255]

    fibrous_lig_mask, atheroma_mask, lumen_mask, none = \
        apply_masks_deformation(fibrous_lig_mask, atheroma_mask, lumen_mask, None)

    mask = plaque_mask.copy()
    mask[mask == 255] = 1
    mask[(fibrous_lig_mask == 255) & (plaque_mask == 255)] = 2
    mask[(atheroma_mask == 255) & (plaque_mask == 255)] = 3
    mask[(lumen_mask == 255) & (plaque_mask == 255)] = 5
    mask[plaque_contour_mask == 255] = 1

    return bg_img, mask


def apply_masks_deformation(fibrous_lig_mask, atheroma_mask, lumen_mask, calc_mask):
    fibrous_lig_mask = mask_deformation(fibrous_lig_mask, random_from_range(0, 2))
    atheroma_mask = mask_deformation(atheroma_mask, random_from_range(0, 2))
    lumen_mask = mask_deformation(lumen_mask, random_from_range(0, 2))
    if calc_mask is not None:
        calc_mask = mask_deformation(calc_mask, random_from_range(0, 2))

    return fibrous_lig_mask, atheroma_mask, lumen_mask, calc_mask


def mask_deformation(mask, state):
    kernel = random_from_range(2, 8)
    if state == 1:
        mask = morphology.erosion(mask, morphology.disk(kernel, dtype='uint8'))
    elif state == 2:
        mask = morphology.dilation(mask, morphology.disk(kernel, dtype='uint8'))
    return mask

def generate_calcification(plaque_mask):
    num = random_from_range(0, 3)
    result = np.zeros((*get_target_size(), 3), dtype=np.uint8)
    result_mask = np.zeros(get_target_size(), dtype=np.uint8)
    vg_calc_color = get_he_color_base() - 5
    for i in range(0, num):
        mask = get_shape_within(plaque_mask, (20, 60), (20, 60), (0, 90))
        vg_result = get_noise_texture(mask, vg_calc_color, 0.04)
        where = (mask == 255) & (plaque_mask == 255)
        result[where] = vg_result[where]
        result_mask[where] = mask[where]

    return result, result_mask


def generate_lumen_images(vg_bg_img, he_bg_img, plaque_mask):
    mask = get_shape_within(plaque_mask, (15, 120), (15, 120), (-30, 30))

    vg_result = np.zeros((*get_target_size(), 3), dtype=np.uint8)
    he_result = np.zeros((*get_target_size(), 3), dtype=np.uint8)
    vg_result[mask == 255] = vg_bg_img[mask == 255]
    he_result[mask == 255] = he_bg_img[mask == 255]

    where = (mask == 255) & (plaque_mask == 255)
    mask[~where] = 0
    vg_result[~where] = 0
    he_result[~where] = 0
    return vg_result, he_result, mask


def generate_fibrous_lig_images(plaque_mask):
    mask = get_shape_within(plaque_mask, (160, 180), (210, 230), (-10, 10))

    vg_fibrous_color = get_vg_color_base() + 5
    he_fibrous_color = get_he_color_base() + 5

    vg_result = get_noise_texture(mask, vg_fibrous_color, 0.007)
    he_result = get_noise_texture(mask, he_fibrous_color, 0.005)

    where = (mask == 255) & (plaque_mask == 255)
    mask[~where] = 0
    vg_result[~where] = 0
    he_result[~where] = 0
    return vg_result, he_result, mask


def generate_atheroma_images(plaque_mask):
    mask = get_shape_within(plaque_mask, (130, 160), (180, 210), (-30, 30))

    vg_atheroma_color = get_vg_color_base() + 2
    he_atheroma_color = get_he_color_base() + 2

    vg_result = get_noise_texture(mask, vg_atheroma_color, 0.03)
    he_result = get_noise_texture(mask, he_atheroma_color, 0.025)

    where = (mask == 255) & (plaque_mask == 255)
    mask[~where] = 0
    vg_result[~where] = 0
    he_result[~where] = 0
    return vg_result, he_result, mask


def generate_plaque_images():
    mask = np.zeros(get_target_size(), dtype=np.uint8)
    rwidth = random_from_range(160, 190)
    rheight = random_from_range(210, 240)
    cv2.ellipse(mask, (256, 256), (rwidth, rheight), 0, 0, 360, 255, -1)

    vg_plaque_color = get_vg_color_base()
    he_plaque_color = get_he_color_base()

    vg_result = get_noise_texture(mask, vg_plaque_color, 0.1)
    he_result = get_noise_texture(mask, he_plaque_color, 0.08)

    contour_kernel_size = random_from_range(2, 6)
    mask_erosion = morphology.erosion(mask, morphology.disk(contour_kernel_size, dtype='uint8'))

    contour_mask = mask - mask_erosion
    vg_result_contour = np.zeros((*get_target_size(), 3), dtype=np.uint8)
    he_result_contour = np.zeros((*get_target_size(), 3), dtype=np.uint8)
    vg_result_contour[contour_mask == 255] = vg_result[contour_mask == 255]
    he_result_contour[contour_mask == 255] = he_result[contour_mask == 255]

    return vg_result, vg_result_contour, he_result, he_result_contour, mask, contour_mask


def get_noise_texture(mask, color, var):
    texture = np.zeros((*get_target_size(), 3), dtype=np.float)
    texture[:, :, 0] = color[0]
    texture[:, :, 1] = color[1]
    texture[:, :, 2] = color[2]
    texture /= 255

    noisy = skimage.util.random_noise(texture, mode='gaussian', var=var)
    texture = (noisy * 255).astype(np.uint8)
    result = np.zeros((*get_target_size(), 3), dtype=np.uint8)
    result[mask == 255] = texture[mask == 255]
    return result


def get_shape_within(another_shape, width, height, rotation):
    maxy, maxx = np.nonzero(another_shape)
    minx, maxx = np.min(maxx), np.max(maxx)
    miny, maxy = np.min(maxy), np.max(maxy)

    rwidth = random_from_range(*width)
    rheight = random_from_range(*height)
    rpositionx = random_from_range(minx + rwidth, maxx - rwidth)
    rpositiony = random_from_range(miny + rheight, maxy - rheight)

    mask = np.zeros(get_target_size(), dtype=np.uint8)
    random_angle_rotation = random_from_range(*rotation)
    cv2.ellipse(mask, (rpositionx, rpositiony), (rwidth, rheight), random_angle_rotation, 0, 360, 255, -1)
    return mask


def random_from_range(min_range, max_range):
    if min_range == max_range or min_range > max_range:
        return min_range
    return random.randint(min_range, max_range)


def get_background_images():
    bg_img = cv2.imread('hist_background_tile.png')
    bg_img = cv2.resize(bg_img, dsize=get_target_size(), interpolation=cv2.INTER_NEAREST)
    bg_img = smoothen(bg_img, 10)
    vg_bg_img = bg_img + random_from_range(0, 20)
    he_bg_img = bg_img + random_from_range(0, 20)
    return vg_bg_img, he_bg_img


def get_vg_color_base():
    return np.array([218, 112, 214])


def get_he_color_base():
    return np.array([221, 160, 221])


def smoothen(img, dim):
    kernel = np.ones((dim, dim), np.float32) / dim**2
    return cv2.filter2D(img, -1, kernel)


if __name__ == "__main__":
    main()
    exit(0)
