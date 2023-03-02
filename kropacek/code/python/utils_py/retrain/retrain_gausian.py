from __future__ import print_function, division

from utils_py.dataset_utils import get_pre_processing

from deeplearning.loader.DatasetWeighted2 import DatasetWeighted2
import time

import albumentations as alb

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from utils_py.dataset_utils import load_text, save_logs, save_learning_graph
from deeplearning.model.modelinput import ModelInput


def main():
    data_dir = "D:\\bakalarka\\to_gaussian"

    train_names = load_text(data_dir + '\\train.txt')
    valid_names = load_text(data_dir + '\\valid.txt')

    pre_processing = get_pre_processing("resnet34", "imagenet")
    dataset_train = DatasetWeighted2(
        data_path=data_dir,
        file_names=train_names,
        augmentation=alb.load("D:\\bakalarka\\to_gaussian\\aug01.json"),
        preprocessing=pre_processing
    )
    dataset_valid = DatasetWeighted2(
        data_path=data_dir,
        file_names=valid_names,
        preprocessing=pre_processing
    )

    # Dataloader iterators, make sure to shuffle
    train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=8, shuffle=True)


    model = torch.load("D:\\bakalarka\\to_gaussian\\weights_black.pth")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.segmentation_head[0] = torch.nn.Conv2d(256, 9, kernel_size=(1, 1), stride=(1, 1))
    model.segmentation_head[1] = torch.nn.UpsamplingBilinear2d(scale_factor=4.0)
    model.segmentation_head[2] = torch.nn.Softmax(dim=1)
    model.segmentation_head.training = True
    model = model.to(device)

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=smp.utils.losses.DiceLoss(),
        metrics=[smp.utils.metrics.Fscore(), smp.utils.metrics.Accuracy()],
        optimizer=torch.optim.Adam([
            {'params': model.encoder.parameters(), 'lr': 10 ** -4},
            {'params': model.decoder.parameters(), 'lr': 10 ** -4},
        ]),
        device="cuda",
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=smp.utils.losses.DiceLoss(),
        metrics=[smp.utils.metrics.Fscore(), smp.utils.metrics.Accuracy()],
        device="cuda",
    )

    # epochs
    all_train_logs = []
    all_valid_logs = []
    max_score = 0
    last_epoch = 0
    start_time = time.time()
    for i_epoch in range(800):

        print('\nEpoch: {}'.format(i_epoch))

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        all_train_logs.append(train_logs)
        all_valid_logs.append(valid_logs)

        if max_score < valid_logs['fscore']:
            max_score = valid_logs['fscore']
            torch.save(model, data_dir + '\\weights_gausian.pth')
            last_epoch = i_epoch
            print('Model saved!')

    end_time = time.time()
    spent_time = end_time - start_time
    print('finished training...')
    save_logs(all_train_logs, all_valid_logs, data_dir)
    print('logs saved...')
    # save_learning_graph(all_train_logs, all_valid_logs, 'Accuracy', 'objective loss', 'learning_curve_loss',
    #         model_input.n_epochs, model_input.model_path)
    save_learning_graph(all_train_logs, all_valid_logs, 'fscore', 'metric score', 'learning_curve_metric',
                        800, data_dir)

    return spent_time, last_epoch


if __name__ == "__main__":
    main()
