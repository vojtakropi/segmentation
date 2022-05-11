import segmentation_models_pytorch as smp

from evaluate_model import evaluate_model
from deeplearning.model.modelinput import ModelInput
from train_model import train_model


def main():
    model_input = ModelInput(evaluate_only=False,
                             model_id='deeplabv3+_02',
                             histology_segmentation_root_str='C:/bakalarka/he',
                             dataset_name='HE',
                             sub_dataset_path_str='dataset_train63_valid20_test0',
                             augmentation_file_name='aug01',
                             model_name='deeplabv3+',
                             encoder_name='resnet34',
                             encoder_weights='imagenet',
                             activation='softmax',
                             in_channels=3,
                             batch_size=8,
                             n_epochs=800,
                             loss_function=smp.utils.losses.DiceLoss(),
                             metrics=[smp.utils.metrics.Fscore(), smp.utils.metrics.Accuracy()],
                             log_lre=-4,
                             log_lrd=-4,
                             )
    model_input.save()
    model_input.init_model()

    if not model_input.evaluate_only:
        spent_time, last_epoch = train_model(model_input)
        evaluate_model(model_input, spent_time, last_epoch)
    else:
        evaluate_model(model_input)


if __name__ == '__main__':
    main()
