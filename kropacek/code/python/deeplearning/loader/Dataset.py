import cv2.cv2 as cv2
import numpy as np
from torch.utils.data import Dataset as DatasetTorch


class Dataset(DatasetTorch):

    def __init__(
            self,
            data_path,
            file_names,
            classes,
            class_indices,

            augmentation=None,
            preprocessing=None,
    ):
        self.data_path = data_path
        self.file_names = file_names
        self.classes = classes
        self.class_indices = class_indices

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        img_name = self.file_names[i]
        img_path = self.data_path / 'images' / img_name
        label_path = self.data_path / 'labels' / img_name
        image = cv2.imread(str(img_path))
        label = cv2.imread(str(label_path), 0)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=label)
            image = sample['image']
            label = sample['mask']

        mask = label2mask(label, len(self.classes), self.class_indices)

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.file_names)


def label2mask(label, n_classes, class_indices):
    mask = np.zeros((label.shape[0], label.shape[1], n_classes), dtype='uint8')
    for i in range(n_classes):
        class_idx = class_indices[i]
        mask[:, :, i] = (label == int(class_idx))
    return mask
