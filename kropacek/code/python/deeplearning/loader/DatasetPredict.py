import cv2
from torch.utils.data import Dataset as DatasetTorch


class DatasetPredict(DatasetTorch):

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
        img_path = self.data_path + "\\"+ img_name
        image = cv2.imread(str(img_path) + '.png')

        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
        return image

    def __len__(self):
        return len(self.file_names)
