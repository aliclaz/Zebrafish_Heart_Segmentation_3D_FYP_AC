# coding: utf-8

if __name__ == '__main__':
    import os

    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_use)

from imgPreprocessing import get_hpf
from keras.models import load_model
from predict_module import predict
from display import show_pred_masks
from statistical_analysis.df_manipulation import gm_df_calcs, healthy_df_calcs, add_df_calcs, add_healthy_df_calcs

IN_PATH = input('What is the directory path to the image(s) you would like to analyse?\n')
OUT_PATH = input('What is the directory path you would like to save the results in?\n')
REUSE = input('Are you adding more images from the same experiment?\n')
STAT_PATH = '.Data/Stats/'
MOD_PATH = '.Data/Models/'
ENTERED_HPF = input('What stage of development were the provided images taken in?\n')
GM = input('What genetic modification has been applied to the embryo in the image(s)?\n')
SCALES = []
while SCALE != 'Done':
    SCALE = input('What is the height/width of the image in micrometres?\nIf all images have the same value, only enter it once.\n Enter Done when this has been completed for all images entered\n')
    SCALES.append(SCALES)

# Get one of the default hpf values (30, 36, 48) based on which the entered value is closest to

mod_hpf = get_hpf(ENTERED_HPF)

# Define the required size parameters of the image for the model and the backbone used in the model

HEIGHT = 512
WIDTH = 512
DEPTH = 512
CHANNELS = 3
BACKBONE = 'resnet34'

def main(args):

    # Load the model being used to make the predictions

    model = load_model(MOD_PATH+'{}HPF_AttentionResUnet_100epochs.h5'.format(mod_hpf), compile=True)

    # Import images, preprocess, make predictions and save the predicted masks

    imgs, preds = predict(model, BACKBONE, IN_PATH, OUT_PATH)

    # Plot the test images and their predicted masks at 3 random slice
    
    show_pred_masks(imgs, preds, OUT_PATH)

    # Find volume of each class in the image
    # Get the class labels for each stage of development

    if mod_hpf == 30:
        classes = ['Background', 'Noise', 'Endocardium', 'Linear Heart Tube']
    elif mod_hpf == 36:
        classes = ['Background', 'Noise', 'Endocardium', 'Atrium', 'Ventricle']
    elif mod_hpf == 48:
        classes = ['Background', 'Noise', 'Endocardium', 'Atrium', 'AVC', 'Ventricle']
        
    # Complete calculations of the volumes of each class from the predicted masks and complete a statistical test as to whether the difference between
    # the mean volume of each class of the entered images is statistically significantly different to that of the mean healthy volume of each class

    # The results will be displayed in the terminal and saved as a CSV file in the user chose output path for access at a later date

    if REUSE == 'yes' or 'Yes' or 'YES' or 'y':
        if GM == 'healthy' or 'Healthy' or 'HEALTHY' or 'None' or 'none' or 'NONE':
            add_healthy_df_calcs(preds, classes, ENTERED_HPF, SCALES, STAT_PATH, OUT_PATH)
        else:
            add_df_calcs(preds, classes, ENTERED_HPF, GM, SCALES, STAT_PATH, OUT_PATH)

    else:
        if GM == 'healthy' or 'Healthy' or 'HEALTHY' or 'None' or 'none' or 'NONE':
            add_healthy_df_calcs(preds, classes, ENTERED_HPF, SCALES, STAT_PATH, OUT_PATH)
        else:
            gm_df_calcs(preds, classes, ENTERED_HPF, GM, SCALES, STAT_PATH, OUT_PATH)