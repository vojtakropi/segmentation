import json
from pathlib import Path

import torch
import cv2.cv2 as cv2
import numpy as np
import pandas as pd
import sklearn.metrics as sm

from loader.DatasetPredict import DatasetPredict
from utils.dataset_utils import load_text
from utils.image_utils import get_visible_label_max


def main():
    model_base_path = Path('C:/bakalarka/vg/datasets/VG/dataset_train71_valid20_test0/results/deeplabv3+_02')

    output_path = Path(model_base_path / 'simulations/predictions_valid')
    output_path.mkdir(parents=True, exist_ok=True)

    model_input = torch.load(model_base_path / 'model_settings.pth')
    model = torch.load(model_base_path / 'weights.pth')

    with open(model_base_path / 'valid_logs.json') as f:
        data = json.load(f)
        data = sorted(data, key=lambda item: item['fscore'], reverse=True)
        print(data[0])

    # by filename
    # img_names = [os.path.splitext(filename)[0] for filename in os.listdir(images_path)]
    img_names = load_text(model_input.dataset_path / 'images_unlabeled.txt')

    test_dataset = DatasetPredict(
        data_path=model_input.dataset_base_path,
        file_names=img_names,
        preprocessing=model_input.pre_processing
    )

    scores_matrix_micro = []
    scores_matrix_macro = []
    # own metrics evaluation, possible to extract predictions
    for index, data in enumerate(test_dataset):
        inputs = data
        name = test_dataset.file_names[index]
        #input_img = labels.transpose(1, 2, 0)
        transformed = torch.from_numpy(inputs).to(model_input.device).unsqueeze(0)
        outputs = model(transformed)
        predicted_img = outputs.cpu().detach().numpy().squeeze().round().transpose(1, 2, 0)
        maximum_per_class = np.ones(predicted_img.shape, dtype=bool)
        for dim in range(0, predicted_img.shape[2]):
            img1 = predicted_img[:, :, dim]
            saved_res = maximum_per_class[:, :, dim]
            for dim2 in range(0, predicted_img.shape[2]):
                if dim != dim2:
                    img2 = predicted_img[:, :, dim2]
                    subres = img1 > img2
                    saved_res = np.bitwise_and(saved_res, subres)
            maximum_per_class[:, :, dim] = saved_res

        result = get_visible_label_max(maximum_per_class)
        cv2.imwrite(str(output_path / (name + '.png')), result)

        scores_micro = []
        scores_raw_micro = []
        scores_macro = []
        scores_raw_macro = []

    print("done")


if __name__ == '__main__':
    main()
    exit(0)
