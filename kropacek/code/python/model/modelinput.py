from pathlib import Path

import torch
import random
import numpy as np
import segmentation_models_pytorch as smp

from utils_py.dataset_utils import get_pre_processing, load_text


class ModelInput:
    def __init__(self, evaluate_only,
                 model_id,
                 histology_segmentation_root_str,
                 dataset_name,
                 sub_dataset_path_str,
                 augmenation_file_name,
                 model_name,
                 encoder_name,
                 encoder_weights,
                 activation,
                 in_channels,
                 batch_size,
                 n_epochs,
                 loss_function,
                 metrics,
                 log_lre,
                 log_lrd,
                 ):
        self.evaluate_only = evaluate_only
        self.sub_dataset_path = Path(sub_dataset_path_str)

        self.model_id = model_id
        self.dataset_name = dataset_name
        self.augmentation_file_name = augmenation_file_name

        self.init_base_dependent_paths(histology_segmentation_root_str)

        self.batch_size = batch_size

        self.model_name = model_name
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.activation = activation
        self.in_channels = in_channels

        self.n_epochs = n_epochs
        self.loss_function = loss_function
        self.metrics = metrics
        self.log_lre = log_lre
        self.log_lrd = log_lrd
        # device
        self.device = 'cuda'
        # seed
        self.seed = 0
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # torch setup
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # init as empty for save purposes
        self.optimizer = {}
        self.model = {}

        self.pre_processing = get_pre_processing(encoder_name, encoder_weights)

    def init_base_dependent_paths(self, histology_segmentation_root_str):
        self.histology_segmentation_root = Path(histology_segmentation_root_str)
        self.dataset_base_path = self.histology_segmentation_root / 'datasets' / self.dataset_name
        self.dataset_path = self.dataset_base_path / self.sub_dataset_path
        # result folders creation
        self.results_path = self.dataset_path / 'results'
        self.results_path.mkdir(exist_ok=True)
        # find augmentations
        self.augmentation_file_path = self.histology_segmentation_root / 'augmentations' / \
                                      (self.augmentation_file_name + '.json')
        # find classes
        self.classes = load_text(self.dataset_base_path / 'classes.txt')
        # model specific folder creation
        self.model_path = self.results_path / self.model_id
        self.model_path.mkdir(exist_ok=True)

    def init_model(self):
        models = {
            'unet': smp.Unet,
            'linknet': smp.Linknet,
            'deeplabv3': smp.DeepLabV3,
            'deeplabv3+': smp.DeepLabV3Plus,
            'fpn': smp.FPN,
            'pan': smp.PAN,
            'pspnet': smp.PSPNet,
        }
        # model init
        self.model = models[self.model_name](
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            classes=len(self.classes),
            activation=self.activation,
            in_channels=self.in_channels
        )

        self.optimizer = torch.optim.Adam([
            {'params': self.model.encoder.parameters(), 'lr': 10 ** self.log_lre},
            {'params': self.model.decoder.parameters(), 'lr': 10 ** self.log_lrd},
        ])

    def save(self):
        torch.save(self, self.model_path / 'model_settings.pth')
