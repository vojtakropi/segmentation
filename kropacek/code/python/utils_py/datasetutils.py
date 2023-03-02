import json

import albumentations as alb
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_pre_processing_fn(encoder_name, encoder_weights):
    return smp.encoders.get_preprocessing_fn(
        encoder_name=encoder_name,
        pretrained=encoder_weights
    )


def get_pre_processing(encoder_name, encoder_weights):
    pre_processing_fn = get_pre_processing_fn(encoder_name, encoder_weights)
    _transform = [
        alb.Lambda(image=pre_processing_fn),
        alb.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return alb.Compose(_transform)


def load_text(path):
    return path.read_text().split('\n')


def load_colors(path):
    return np.array(eval(path.read_text().replace('\n', ','))).astype('uint8')


def save_logs(all_train_logs, all_valid_logs, results_path):
    with open(results_path / 'train_logs.json', 'w') as train_logs_file:
        json.dump(all_train_logs, train_logs_file)
    with open(results_path / 'valid_logs.json', 'w') as valid_logs_file:
        json.dump(all_valid_logs, valid_logs_file)


def save_learning_graph(all_train_logs, all_valid_logs, metric, title, graph_name, n_epochs, results_path):
    train_losses = [all_train_logs[e][metric] for e in range(n_epochs)]
    valid_losses = [all_valid_logs[e][metric] for e in range(n_epochs)]

    plt.figure()
    plt.plot(range(n_epochs), train_losses, label='train')
    plt.plot(range(n_epochs), valid_losses, label='valid')
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.legend(loc='best')
    plt.title(title)
    plt.savefig(results_path / (graph_name + '.png'), dpi=200)