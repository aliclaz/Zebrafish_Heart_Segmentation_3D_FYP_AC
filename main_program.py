import argparse

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

# Define the input paths for the trained model and the csv files containing the data from previous statistical analysis

STAT_PATH = '.Data/Stats/'
MOD_PATH = '.Data/Models/'

# Define the required size parameters of the image for the model and the backbone used in the model

HEIGHT = 512
WIDTH = 512
DEPTH = 512
BACKBONE = 'resnet34'

def main(args):

    # Load the model being used to make the predictions

    model = load_model(MOD_PATH+'{}HPF_AttentionResUnet_100epochs.h5'.format(mod_hpf), compile=True)

    # Import images, preprocess, make predictions and save the predicted masks

    imgs, preds = predict(model, BACKBONE, args.args.args.in_path, args.out_path)

    # Plot the test images and their predicted masks at 3 random slice
    
    show_pred_masks(imgs, preds, args.out_path)

    
    # Get one of the default hpf values (30, 36, 48) based on which the entered value is closest to

    mod_hpf = get_hpf(args.entered_hpf)


    scales = args.scale.split(' ')
    scales = [int(scale) for scale in scales]

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

    if args.reuse == 'yes' or 'Yes' or 'YES' or 'y':
        if args.gm == 'healthy' or 'Healthy' or 'HEALTHY' or 'None' or 'none' or 'NONE':
            add_healthy_df_calcs(preds, classes, args.entered_hpf, scales, STAT_PATH, args.out_path)
        else:
            add_df_calcs(preds, classes, args.entered_hpf, args.gm, scales, STAT_PATH, args.out_path)

    else:
        if args.gm == 'healthy' or 'Healthy' or 'HEALTHY' or 'None' or 'none' or 'NONE':
            add_healthy_df_calcs(preds, classes, args.entered_hpf, scales, STAT_PATH, args.out_path)
        else:
            gm_df_calcs(preds, classes, args.entered_hpf, args.gm, scales, STAT_PATH, args.out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_path', type=str, help='What is the directory path to the image(s) you would like to analyse?', required=True)
    parser.add_argument('--out_path', type=str, help='What is the directory path you would like to save the results in?', required=True)
    parser.add_argument('--reuse', type=str, help='Are you adding more images from the same experiment? (Yes or No)', required=True)
    parser.add_argument('--entered_hpf', type=int, help='What stage of development, in hours post-fertilisation (hpf), were the provided images taken in?', required=True)
    parser.add_argument('--gm', type=int, help='What genetic modification was applied to the zebrafish embryo prior to the image being taken?', required=True)
    parser.add_argument('--scale', type=str, help='Enter the width dimension of each image in \u03bcm separated by a single space. Press ENTER when done.', required=True)

    args = parser.parse_args()

    main(args)