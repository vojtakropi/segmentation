import cv2
import tifffile
from torch.utils.data import Dataset as DatasetTorch
from numpy import asarray

class DatasetWeighted(DatasetTorch):

    def __init__(
            self,
            data_path,
            file_names,

            augmentation=None,
            preprocessing=None,
    ):
        self.data_path = data_path
        self.file_names = file_names
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        img_name = self.file_names[i]
        img_path = self.data_path + '\\ultr_resized\\' + img_name
        label_path = self.data_path + '\\ultr_resized\\' + img_name
        image = cv2.imread(str(img_path) + '.png')
        label = cv2.imread(str(label_path) + '.png')

        label = label / 255
        label = asarray(label)
        if self.augmentation:
            sample = self.augmentation(image=image, mask=label)
            image = sample['image']
            label = sample['mask']

        mask = label

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask

    def __len__(self):
        return len(self.file_names)
