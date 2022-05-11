from pathlib import Path

import torch

from deeplearning.model.modelinput import ModelInput


def main():
    model_base_path = Path(
        'C:/bakalarka/he/datasets/HE/dataset_train337_valid84_test0/results/deeplabv3+_02')

    model_input = torch.load(model_base_path / 'model_settings.pth')
    model_input.dataset_name = 'HE'
    model_input.augmentation_file_name = 'aug01.json'
    model_input.model_id = 'deeplabv3+_02'
    model_input.init_base_dependent_paths('C:/bakalarka/he')

    model_input_dst = ModelInput(model_input.evaluate_only,
                                 model_input.model_id,
                                 str(model_input.histology_segmentation_root),
                                 model_input.dataset_name,
                                 str(model_input.sub_dataset_path),
                                 model_input.augmentation_file_name,
                                 model_input.model_name,
                                 model_input.encoder_name,
                                 model_input.encoder_weights,
                                 model_input.activation,
                                 model_input.in_channels,
                                 model_input.batch_size,
                                 model_input.n_epochs,
                                 model_input.loss_function,
                                 model_input.metrics,
                                 model_input.log_lre,
                                 model_input.log_lrd)

    model_input_dst.save()


if __name__ == "__main__":
    main()
