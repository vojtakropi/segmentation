import time

import albumentations as alb

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from loader.DatasetWeighted import DatasetWeighted
from utils.dataset_utils import load_text, save_logs, save_learning_graph


def train_model(model_input):

    train_names = load_text(model_input.dataset_path / 'train.txt')
    valid_names = load_text(model_input.dataset_path / 'valid.txt')

    train_dataset = DatasetWeighted(
        data_path=model_input.dataset_base_path,
        file_names=train_names,
        augmentation=alb.load(model_input.augmentation_file_path),
        preprocessing=model_input.pre_processing
    )

    valid_dataset = DatasetWeighted(
        data_path=model_input.dataset_base_path,
        file_names=valid_names,
        preprocessing=model_input.pre_processing
    )

    train_loader = DataLoader(train_dataset, batch_size=model_input.batch_size, shuffle=True, num_workers=8)

    valid_loader = DataLoader(valid_dataset, batch_size=model_input.batch_size, shuffle=False, num_workers=8)

    train_epoch = smp.utils.train.TrainEpoch(
        model_input.model,
        loss=model_input.loss_function,
        metrics=model_input.metrics,
        optimizer=model_input.optimizer,
        device=model_input.device,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model_input.model,
        loss=model_input.loss_function,
        metrics=model_input.metrics,
        device=model_input.device,
    )

    # epochs
    all_train_logs = []
    all_valid_logs = []
    max_score = 0
    last_epoch = 0
    start_time = time.time()
    for i_epoch in range(model_input.n_epochs):

        print('\nEpoch: {}'.format(i_epoch))

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        all_train_logs.append(train_logs)
        all_valid_logs.append(valid_logs)

        if max_score < valid_logs['fscore']:
            max_score = valid_logs['fscore']
            torch.save(model_input.model, model_input.model_path / 'weights.pth')
            last_epoch = i_epoch
            print('Model saved!')

    end_time = time.time()
    spent_time = end_time - start_time
    print('finished training...')
    save_logs(all_train_logs, all_valid_logs, model_input.model_path)
    print('logs saved...')
   # save_learning_graph(all_train_logs, all_valid_logs, 'Accuracy', 'objective loss', 'learning_curve_loss',
               #         model_input.n_epochs, model_input.model_path)
    save_learning_graph(all_train_logs, all_valid_logs, 'fscore', 'metric score', 'learning_curve_metric',
                        model_input.n_epochs, model_input.model_path)

    return spent_time, last_epoch
