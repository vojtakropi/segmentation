import json
import time

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import sklearn.metrics as sm
import torch
from torch.utils.data import DataLoader

from loader.DatasetWeighted import DatasetWeighted
from utils.dataset_utils import load_text


def evaluate_model(model_input, spent_time=-1, last_epoch=-1):
    print('test set evaluation...')
    model = torch.load(model_input.model_path / 'weights.pth')
    test_names = load_text(model_input.dataset_path / 'valid.txt')

    test_dataset = DatasetWeighted(
        data_path=model_input.dataset_base_path,
        file_names=test_names,
        preprocessing=model_input.pre_processing
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    # test evaluation
    test_epoch = smp.utils.train.ValidEpoch(
        model=model,
        loss=model_input.loss_function,
        metrics=model_input.metrics,
        device=model_input.device,
    )

    test_logs = test_epoch.run(test_loader)
    loss_function_name = model_input.loss_function.__name__
    performance = {
        'dice': np.round(100 * test_logs['fscore'], decimals=1),
        loss_function_name: round(test_logs[loss_function_name], 2),
        'training_time': time.strftime("%Hh %Mm %Ss", time.gmtime(spent_time)),
        'last_epoch': last_epoch
    }
    with open(model_input.model_path / 'test_performance.json', 'w') as f:
        json.dump(performance, f)

    # own metrics evaluation, possible to extract predictions
    f1_per_img = {}
    scores_matrix = []
    for index, data in enumerate(test_dataset):
        inputs, labels = data
        name = test_dataset.file_names[index]
        input_img = labels.transpose(1, 2, 0)
        transformed = torch.from_numpy(inputs).to(model_input.device).unsqueeze(0)
        outputs = model(transformed)
        predicted_img = outputs.cpu().detach().numpy().squeeze().round().transpose(1, 2, 0)
        scores = []
        scores_raw = []
        for i in range(0, len(model_input.classes)):
            score = sm.f1_score(input_img[:, :, i].ravel(), predicted_img[:, :, i].ravel())
            scores_raw.append(score)
            scores.append({model_input.classes[i]: score})
        f1_per_img[name] = scores
        scores_matrix.append(scores_raw)

    scores_data_frame = pd.DataFrame(
        np.array(scores_matrix),
        index=test_names,
        columns=model_input.classes
    ).sort_index()

    scores_data_frame.to_excel(model_input.model_path / 'f1_per_image_per_class.xlsx')
