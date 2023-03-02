import os

import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt

from utils_py.image_utils import segment2mask


def create_patients_structure():
    imgs = os.listdir(imgs_dir)
    patients = {}
    for img in imgs:
        img_name = imgs_dir + "/" + img
        img_name_split = img.split('_')
        patient_id = img_name_split[0]
        seq_id = "_".join(img_name_split[1:3])
        seq_img_key = seq_imgs_key_name
        if 'label' in img_name:
            seq_img_key = seq_labels_key_name
        patient = {}
        if patient_id not in patients:
            patients[patient_id] = patient
        else:
            patient = patients[patient_id]
        seq = {}
        if seq_id not in patient:
            patient[seq_id] = seq
        else:
            seq = patient[seq_id]
        seq_imgs = []
        if seq_img_key not in seq:
            seq[seq_img_key] = seq_imgs
        else:
            seq_imgs = seq[seq_img_key]
        seq_imgs.append(img)
    return patients


def merge_images(patients):
    for patient_key in patients.keys():
        patient = patients[patient_key]
        for seq_key in patient.keys():
            seq = patient[seq_key]
            # does not contain labels
            if len(seq.keys()) == 1:
                continue
            imgs = seq[seq_imgs_key_name]
            labels = seq[seq_labels_key_name]
            # less labels than images
            if len(imgs) == 0:
                continue
            if len(imgs) != 2:
                print('patient: ' + patient_key + ', seq: ' + seq_key + ' does not contain 2 images')

            print('processing patient: ' + patient_key + ', seq: ' + seq_key)
            he = next((name for name in imgs if 'HE' in name), '')
            vg = next((name for name in imgs if 'VG' in name), '')

            helbl = next((name for name in labels if 'HE' in name), '')
            vglbl = next((name for name in labels if 'VG' in name), '')

            if vglbl is '':
                continue
            he_img = cv2.imread(imgs_dir + '/' + vg)
            he_img = cv2.resize(he_img, resize, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(output_dir + '/images/' + patient_key + '_' + seq_key + '.png', he_img)
            he_lbl_img = cv2.imread(imgs_dir + '/' + vglbl)
            he_lbl_img = cv2.resize(he_lbl_img, resize, interpolation=cv2.INTER_NEAREST)

            result_label = segment2mask(he_lbl_img, real_colors)
            result_label *= 255

            tifffile.imwrite(output_dir + '/labels/' + patient_key + '_' + seq_key + '.tiff',
                            result_label)


def merge2masks(first_mask, second_mask, result):
    if probabilistic:
        full = cv2.bitwise_and(first_mask, second_mask)
        partial = cv2.bitwise_or(first_mask, second_mask) - full
        result[full == 1] = 1
        result[partial == 1] = 0.5
    else:
        partial = cv2.bitwise_or(first_mask, second_mask)
        result[partial == 1] = 1


if __name__ == '__main__':
    imgs_dir = "D:\\bakalarka\\all_images"
    output_dir = 'D:\\bakalarka\\tiff_test'
    seq_imgs_key_name = 'imgs'
    seq_labels_key_name = 'labels'
    probabilistic = True
    export_single_imgs = False
    # we ignore 6,7,8,10,11,12
    mask_values = (
        0, 1, 2,
        3, 4, 5,
        9, 13, 14
    )
    # here not RGB but BGR because of OPENCV.
    real_colors = (
        (0, 0, 0), (0, 0, 255), (0, 255, 0),
        (255, 0, 0), (0, 0, 85), (0, 170, 0),
        (255, 0, 255), (185, 255, 30), (128, 128, 128)
    )

    plaque_color = (185, 255, 30)

    resize = (512, 512)
    patients = create_patients_structure()
    merge_images(patients)
    exit(0)
