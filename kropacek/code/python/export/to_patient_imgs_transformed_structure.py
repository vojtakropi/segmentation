import scipy.io
import os
import numpy as np
from pathlib import Path
import cv2

if __name__ == '__main__':
    working_dir = "F:/ondra/ateroskleroza-data/histology_segmentation/data/patient_to_img"
    transformations_dir = 'F:/ondra/ateroskleroza-data/histology_segmentation/data/hist1049_pacient138_stainHEVG_Paired2_rotatedmats'
    imgs_dir = 'F:/ondra/ateroskleroza-data/histology_segmentation/data/hist1049_pacient138_stainHEVG'
    output_dir = "F:/ondra/ateroskleroza-data/histology_segmentation/data/patient_to_img_transformed"
    dirs = os.listdir(transformations_dir)
    for dir_name in dirs:
        curr_dir = working_dir + '/' + dir_name
        imgs_curr_dir = imgs_dir + '/' + dir_name
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        patient_transformations_dir = transformations_dir + '/' + dir_name + '/' + 'finalmanual'
        patient_transformations = os.listdir(patient_transformations_dir)
        transformation_matrices = set()
        transformation_imgs = set()
        for transformation_file in patient_transformations:
            if '.mat' in transformation_file:
                transformation_matrices.add(transformation_file.replace('.mat', ''))
            elif '.png' in transformation_file:
                transformation_imgs.add(transformation_file.replace('.png', ''))
        reference_imgs = transformation_imgs - transformation_matrices
        for reference_img_full_name in reference_imgs:
            reference_prefix = reference_img_full_name[0]
            reference_img_name = reference_img_full_name[4:]
            reference_img = cv2.imread(imgs_curr_dir + '/' + reference_img_name.replace('_', '/') + '.png')
            cv2.imwrite(output_dir + '/' + dir_name + '_' + reference_img_full_name + '.png', reference_img)
            reference_img_label = cv2.imread(curr_dir + '/' + reference_img_name + '_label.png')
            if reference_img_label is None:
                print('reference image is not labeled in annotated dataset')
            else:
                cv2.imwrite(output_dir + '/' + dir_name + '_' + reference_img_full_name + '_label.png',
                            reference_img_label)
            for matrix in transformation_matrices:
                if matrix.startswith(reference_prefix):
                    matrix_dir = patient_transformations_dir + '/' + matrix + '.mat'
                    original_file_name = matrix[7:] + '.png'

                    to_transform_img_path = imgs_curr_dir + '/' + matrix + '.png'
                    to_transform_img = cv2.imread(imgs_curr_dir + '/' + matrix[4:].replace('_', '/') + '.png')

                    to_transform_label_path = curr_dir + '/' + matrix[4:] + '_label.png'
                    to_transform_label = cv2.imread(to_transform_label_path)

                    cols = reference_img.shape[0]
                    rows = reference_img.shape[1]

                    transformation_matrix_ml_structure = scipy.io.loadmat(
                        patient_transformations_dir + '/' + matrix + '.mat')
                    if 'matice' in transformation_matrix_ml_structure.keys():
                        transformation_matrix_raw = transformation_matrix_ml_structure['matice']
                    elif 'Ftrans' in transformation_matrix_ml_structure.keys():
                        transformation_matrix_raw = transformation_matrix_ml_structure['Ftrans']
                    else:
                        raise BaseException('no transformation matrix present in data')
                    transformation_matrix = np.transpose(transformation_matrix_raw)[:-1, :]
                    if to_transform_img is not None:
                        to_transform_img = cv2.resize(to_transform_img, (cols, rows), interpolation=cv2.INTER_NEAREST)
                        to_transform_img = cv2.warpAffine(to_transform_img, transformation_matrix, (cols, rows),
                                                          flags=cv2.INTER_NEAREST,
                                                          borderMode=cv2.BORDER_REPLICATE)
                        cv2.imwrite(output_dir + '/' + dir_name + '_' + matrix + '.png', to_transform_img)
                    if to_transform_label is None:
                        print('transformed image is not annotated, skipping...')
                    else:
                        to_transform_label = cv2.resize(to_transform_label, (cols, rows),
                                                        interpolation=cv2.INTER_NEAREST)
                        to_transform_label = cv2.warpAffine(to_transform_label, transformation_matrix, (cols, rows),
                                                            flags=cv2.INTER_NEAREST,
                                                            borderMode=cv2.BORDER_REPLICATE)
                        cv2.imwrite(output_dir + '/' + dir_name + '_' + matrix + '_label.png', to_transform_label)

                    print(to_transform_img_path)

    print("run script done!")
    exit(0)
