import numpy as np
import cv2
import os


def img_to_colored(gray):
    rgb_clean = np.stack(np.stack((gray,) * 3, -1))
    visible_mask = np.zeros(rgb_clean.shape, 'uint8')
    for i in mask_values:
        visible_mask[np.where((rgb_clean == i).all(-1))] = real_colors[i]
    return visible_mask


if __name__ == '__main__':
    working_dir = "C:/projects/phd/data/ateroskleroza/svetla_tmava_imgs/svetla/data"
    output_file_prefix = "all_masks_partial"
    mask_values = (
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        9, 10, 11,
        12, 13, 14
    )
    # here not RGB but BGR because of OPENCV.
    real_colors = (
        (0, 0, 0), (0, 0, 255), (0, 255, 0),
        (255, 0, 0), (0, 0, 85), (0, 170, 0),
        (127, 0, 255), (255, 255, 0), (0, 85, 0),
        (255, 0, 255), (0, 85, 255), (0, 165, 255),
        (0, 255, 255), (185, 255, 30), (128, 128, 128)
    )

    processed_masks = (
        1, 2, 3, 4, 5, 9, 14
    )

    dirs = os.listdir(working_dir)
    for dir_name in dirs:
        curr_dir = working_dir + "/" + dir_name
        masks = curr_dir + "/" + "masks/preprocessed"
        if not os.path.exists(masks):
            print("masks folder does not exist in: " + curr_dir)
            continue
        mask_names = os.listdir(masks)
        imgs = {}
        shape = None
        for mask_name in mask_names:
            prefix = mask_name.split("_")[0]
            if output_file_prefix.find(prefix) > -1:
                continue
            mask_original_value = int(prefix)
            if mask_original_value not in processed_masks:
                continue
            img = cv2.cvtColor(cv2.imread(masks + "/" + mask_name), cv2.COLOR_RGB2GRAY)
            # change 14 for 13 (
            if mask_original_value == 14:
                mask_original_value = 13
            # bin for sure
            img[img > 0] = 255
            img[img == 255] = mask_original_value
            if shape is None:
                shape = img.shape
            imgs[mask_original_value] = img
        if len(imgs.keys()) > 0:
            canvas = np.zeros(shape, "uint8")
            area_sorted = {k: v for k, v in sorted(imgs.items(), key=lambda item: np.count_nonzero(item[1]), reverse=True)}
            for mask_value in area_sorted.keys():
                mask = area_sorted[mask_value]
                where = mask == mask_value
                canvas[where] = mask[where]
            cv2.imwrite(masks + "/" + output_file_prefix + "original.png", canvas)
            rgb = img_to_colored(canvas)
            cv2.imwrite(masks + "/" + output_file_prefix + "colored.png", rgb)
            print("carotid masks generated for: " + curr_dir)

    print("run script done!")
    exit(0)


