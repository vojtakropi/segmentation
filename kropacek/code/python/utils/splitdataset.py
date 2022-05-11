import math
from pathlib import Path
import random


def export_dataset(train_filenames, valid_filenames, test_filenames, data_dir):
    dataset_dir_name = f'dataset_train{len(train_filenames)}_valid{len(valid_filenames)}' \
                       f'_test{len(test_filenames)}'
    (data_dir / dataset_dir_name).mkdir(exist_ok=True)
    with open(data_dir / dataset_dir_name / 'train.txt', "w") as f:
        f.write('\n'.join(train_filenames))

    with open(data_dir / dataset_dir_name / 'valid.txt', "w") as f:
        f.write('\n'.join(valid_filenames))

    with open(data_dir / dataset_dir_name / 'test.txt', "w") as f:
        f.write('\n'.join(test_filenames))


def split_dataset(train_ratio, valid_ratio, test_ratio, data_dir):
    data_dir = Path(data_dir)
    file_names = set(sorted([fn.stem for fn in (data_dir / 'images_unlabeled').iterdir()]))
    patients = {}
    for img in file_names:
        img_name_split = img.split('_')
        patient_id = img_name_split[0]
        seq_id = "_".join(img_name_split[1:3])
        if len(img_name_split) >= 4:
            seq_id += '_' + img_name_split[3]
        patient = {'seq': []}
        if patient_id not in patients:
            patients[patient_id] = patient
        else:
            patient = patients[patient_id]
        patient['seq'].append(seq_id)

    n_annotated_patients = len(patients.keys())

    n_train = math.floor(train_ratio * n_annotated_patients)
    n_valid = math.ceil(valid_ratio * n_annotated_patients)
    n_test = n_annotated_patients - n_train - n_valid

    random.seed(4)

    shuffled_keys = list(patients.keys())
    random.shuffle(shuffled_keys)

    train_filenames = []
    valid_filenames = []
    test_filenames = []

    valid_idx = n_train + n_valid
    for i in range(0, n_annotated_patients):
        patient_key = shuffled_keys[i]
        patient = patients[patient_key]
        for p_img in patient['seq']:
            p_img_full = patient_key + '_' + p_img
            if i in range(0, n_train):
                train_filenames.append(p_img_full)
            elif i in range(n_train, valid_idx):
                valid_filenames.append(p_img_full)
            else:
                test_filenames.append(p_img_full)

    export_dataset(train_filenames, valid_filenames, test_filenames, data_dir)


# Used to split dataset so that no patient image is in train/test/valid at the same time
# splitdataset 0.7 0.15 0.15 HEVG c:/projects/ateroskleroza-data/histology_segmentation/datasets/hist175_classes8_size512x512_noMargin_stainHEVG
if __name__ == "__main__":
    # train_ratio = float(sys.argv[1])
    # valid_ratio = float(sys.argv[2])
    # test_ratio = float(sys.argv[3])
    # data_dir = sys.argv[4]
    train_ratio = 0.0
    valid_ratio = 0.0
    test_ratio = 0.0
    data_dir = 'C:/bakalarka/vg/datasets/VG'
    split_dataset(train_ratio, valid_ratio, test_ratio, data_dir)
    exit(0)
