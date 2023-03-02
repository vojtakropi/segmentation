import h5py
import numpy as np
import cv2
import os
from export import img_filler


def img_to_colored(img):
    visible_mask = np.zeros((img.shape[0], img.shape[1], 3), 'uint8')
    for i in mask_values.keys():
        where = np.where((img == i).all(-1))
        # true array is first
        if len(where[0]) > 0:
            mask_values[i] += 1
        visible_mask[where] = real_colors[i]
    return visible_mask


def fix_fill_masks(img):
    mask_colors = np.unique(img)
    shape = img.shape
    empty = np.zeros(shape, 'uint8')
    # count pixels to know filled masks rendering order
    masks_pixel_counts = {}
    for i in mask_colors:
        if i != 0:
            masks_pixel_counts[i] = np.where(img == i)[0].size

    result = empty.copy()
    masks_order = {k: v for k, v in sorted(masks_pixel_counts.items(), key=lambda item: item[1], reverse=True)}
    # fill in rendering order
    for i in masks_order.keys():
        single_mask = empty.copy()
        where = np.where((img == i))
        single_mask[where] = img[where]
        single_mask = img_filler.fill_multi_object_img(single_mask)
        where = np.where((single_mask == i))
        result[where] = single_mask[where]

    return result


def simple_fill(single_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    single_mask = cv2.dilate(single_mask, kernel)
    single_mask = cv2.morphologyEx(single_mask, cv2.MORPH_CLOSE, kernel)
    single_mask = cv2.erode(single_mask, kernel)

    contours, hierarchy = cv2.findContours(single_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    connected_cnt = np.zeros(single_mask.shape, 'uint8')
    for cnt in contours:
        cv2.drawContours(connected_cnt, [cnt], 0, 255, -1)
    return connected_cnt


def process_raw_image_data(item):
    img_name_group_name = item.name
    file_path_group_name = item.get(img_name_group_name + "/Raw Data/filePath")
    # get base image path name
    file_path_group_value = file_path_group_name[()].decode("utf-8")
    # load img
    img = cv2.imread(workingDir + "/" + file_path_group_value)
    nickname_group_name = item.get(img_name_group_name + '/Raw Data/nickname')
    nickname_group_value = nickname_group_name[()].decode("utf-8")
    # get labels data for given image path
    label_data_group = labelSetsList[i]
    label_data_group_blocks_list = list(label_data_group.values())

    # generate canvas for each label
    empty = np.zeros(img.shape[:-1], 'uint8')
    all_labels = empty.copy()
    print("processing file..." + file_path_group_value)
    # create export dir if does not exist
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    # create dir for each img - if exists, do not regenerate
    file_dir = outputDir + "/" + nickname_group_value
    if os.path.exists(file_dir):
        print("directory already exists: " + file_dir)
        return

    os.mkdir(file_dir)
    # get raw data
    process_image_data_block(label_data_group_blocks_list, all_labels)
    # generate ground truth masks base / not filled and folder
    unique = np.unique(all_labels)
    masks_dir = file_dir + "/masks"
    if os.path.exists(masks_dir):
        print("directory already exists: " + masks_dir)
        return
    os.mkdir(masks_dir)
    for u in unique:
        if u != 0:
            single_mask = empty.copy()
            single_mask[all_labels == u] = 255
            cv2.imwrite(masks_dir + "/" + str(u) + "_single_mask.png", single_mask)
            if fill:
                single_result = img_filler.fill_multi_object_img(single_mask)
                cv2.imwrite(masks_dir + "/" + str(u) + "_single_mask_filled.png", single_result)
                single_result = simple_fill(single_mask)
                cv2.imwrite(masks_dir + "/" + str(u) + "_single_mask_filled_simple.png", single_result)

    if transform_mode == "color":
        # make rgb
        all_labels = np.stack(np.stack((all_labels,) * 3, -1))
        all_labels = img_to_colored(all_labels)

    if export_original:
        cv2.imwrite(file_dir + "/" + nickname_group_value + "_original.png", img)

    mask_suffix = "_mask_filled" if fill else "_mask"
    export_file_name = file_dir + "/" + nickname_group_value + mask_suffix + ".png"
    cv2.imwrite(export_file_name, all_labels)
    print("file written to path: " + export_file_name)


def process_image_data_block(label_data_group_blocks_list, all_labels):
    for j in range(len(label_data_group_blocks_list)):
        # for each label data block, we wil generate mask images
        label_data_group_block = label_data_group_blocks_list[j]
        label_data_block_range_value_string = list(label_data_group_blocks_list[j].attrs.values())[0] \
            .decode("utf-8").replace("[", "").replace("]", "")
        delimetered = label_data_block_range_value_string.split(",")
        # parse ranges from string to np arrays
        height_arr = delimetered[0].split(":")
        width_arr = delimetered[1].split(":")
        # reshape to match result matrix
        label_block_data = label_data_group_block[()] \
            .reshape(all_labels[int(height_arr[0]):int(height_arr[1]), int(width_arr[0]):int(width_arr[1])].shape)
        all_labels[int(height_arr[0]):int(height_arr[1]), int(width_arr[0]):int(width_arr[1])] = label_block_data


if __name__ == '__main__':
    workingDir = "C:/projects/phd"
    outputDir = workingDir + "/export-again"
    filename = "Ateroskleroza_vzor_6.ilp"
    export_original = True
    transform_mode = "color"
    fill = True
    mask_values = {
        1: 0, 2: 0,
        3: 0, 4: 0, 5: 0,
        6: 0, 7: 0, 8: 0,
        9: 0, 10: 0, 11: 0,
        12: 0, 13: 0
    }
    # here not RGB but BGR because of OPENCV.
    real_colors = (
        (0, 0, 0), (0, 0, 255), (0, 255, 0),
        (255, 0, 0), (0, 0, 85), (0, 170, 0),
        (127, 0, 255), (255, 255, 0), (0, 85, 0),
        (255, 0, 255), (0, 85, 255), (0, 165, 255),
        (0, 255, 255), (128, 130, 128)
    )

    ilpPath = workingDir + "/" + filename
    print("starting the export " + ilpPath)
    with h5py.File(ilpPath, 'r') as f:
        labelSets = f.get("/PixelClassification/LabelSets")
        labelNames = f.get("/PixelClassification/LabelNames")
        # transpose stored matrix to get normal color map
        pMapColors = f.get("/PixelClassification/PmapColors")[()].T
        rawImageDataGroups = f.get("/Input Data/infos")

        rawImageDataList = list(rawImageDataGroups.values())
        labelSetsList = list(labelSets.values())
        results = list()
        for i in range(0, len(rawImageDataList)):
            process_raw_image_data(rawImageDataList[i])

        print("masks presence in files: " + str(mask_values))
    print("export done ")
    exit(1)
