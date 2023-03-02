import os
import cv2

if __name__ == '__main__':

    working_dir = "C:/projects/phd/data/ateroskleroza/svetla_tmava_imgs/svetla"
    data_folder = working_dir + "/data"
    output_file_prefix = "all_masks_"

    # see readme for mask numbers explanation
    mask_values = (
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        9, 10, 11,
        12, 13, 14
    )
    masks_for_plaque_sub = {1, 2, 3, 4, 5, 6, 7}
    lumen_number = 9

    data_folder_contents = os.listdir(data_folder)
    for single_data_folder in data_folder_contents:
        masks_folder = data_folder + "/" + single_data_folder + "/masks"
        preprocessed_folder = masks_folder + "/preprocessed"
        plaque_path = data_folder + "/" + single_data_folder + "/plaque.png"
        if not os.path.exists(plaque_path):
            print("path " + plaque_path + " does not exist.")
            continue
        plaque = cv2.cvtColor(cv2.imread(plaque_path), cv2.COLOR_RGB2GRAY)
        plaque[plaque == 255] = 1
        data_folder_contents = os.listdir(data_folder)
        mask_names = os.listdir(masks_folder)
        if not os.path.exists(preprocessed_folder):
            os.mkdir(preprocessed_folder)
        for mask_name in mask_names:
            prefix = mask_name.split("_")[0]
            if output_file_prefix.find(prefix) > -1 or mask_name == 'preprocessed':
                continue
            mask_original_value = int(prefix)
            single_mask = cv2.cvtColor(cv2.imread(masks_folder + "/" + mask_name), cv2.COLOR_RGB2GRAY)
            if mask_original_value in masks_for_plaque_sub:
                single_mask[single_mask != 0] = 1
                intersection = single_mask + plaque
                intersection[intersection != 2] = 0
                intersection[intersection == 2] = 255
                cv2.imwrite(preprocessed_folder + "/" + mask_name, intersection)
            elif mask_original_value == lumen_number:
                single_mask[single_mask != 0] = 1
                intersection = single_mask + plaque
                intersection[intersection != 2] = 0
                intersection[intersection == 2] = 1

                intersection = single_mask + intersection
                intersection[intersection != 1] = 0
                intersection[intersection == 1] = 255
                single_mask[single_mask == 1] = 255
                cv2.imwrite(preprocessed_folder + "/9_lumen.png", intersection)
                cv2.imwrite(preprocessed_folder + "/13_unknown_lumen.png", single_mask)
                # original lumen should be UNKNOWN INSIDE LUMEN 13, hole inside label should be LUMEN 9
            else:
                cv2.imwrite(preprocessed_folder + "/" + mask_name, single_mask)
        plaque[plaque == 1] = 255
        cv2.imwrite(preprocessed_folder + "/14_plaque.png", plaque)
        print("preprocessed masks for: " + single_data_folder)

    exit(0)
