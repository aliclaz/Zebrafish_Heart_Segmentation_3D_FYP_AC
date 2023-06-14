import argparse
import tensorflow as tf

# coding: utf-8

if __name__ == '__main__':
    import os
    os.environ['KERAS_BACKEND'] = 'tensorflow'

from imgPreprocessing import get_hpf
from keras.models import load_model
from predict_module import predict
from display import show_pred_masks, disp_3D_pred
from statistical_analysis.df_manipulation import gm_df_calcs, healthy_df_calcs, add_df_calcs, add_healthy_df_calcs

# Define the input paths for the trained model and the csv files containing the data from previous statistical analysis

STAT_PATH = '/Stats/'
MOD_PATH = '/Models/'
OUT_PATH = '/Results/'

# Define the required size parameters of the image for the model and the backbone used in the model

HEIGHT = 512
WIDTH = 512
DEPTH = 512

def main(args):

    # Put the input file names into a list

    in_files = args.in_files.split(',')

    # Get one of the default hpf values (30, 36, 48) based on which the entered value is closest to

    mod_hpf = get_hpf(args.entered_hpf)

    # Import images, preprocess, load_model, make predictions and save the predicted masks

    imgs, preds = predict(MOD_PATH+'{}HPF_{}_{}_{}epochs.h5'.format(mod_hpf, args.backbone, args.model_name, args.epochs), args.model_name, args.backbone, in_files, OUT_PATH, args.entered_hpf, GM=args.gm)

    # Get the class labels for each stage of development

    if mod_hpf == 30:
        classes = ['Background', 'AVC', 'Endocardium' 'Noise', 'Atrium', 'Ventricle']
    elif mod_hpf == 36:
        classes = ['Background', 'Endocardium', 'Atrium', 'Noise', 'Ventricle']
    elif mod_hpf == 48:
        classes = ['Background', 'Noise', 'Endocardium', 'Atrium', 'AVC', 'Ventricle']

    # Plot the test images and their predicted masks at 3 random slice
    
    show_pred_masks(args.model_name, args.backbone, imgs, preds, OUT_PATH, classes)

    # Show the 3D predicted mask for each image

    disp_3D_pred(preds, args.model_name, args.backbone, OUT_PATH, classes, args.entered_hpf)

    # When setting the length scale variable in the bash script, each scale is separated by a space
    # This adds each scale to a list in the form float

    scales = args.scale.split(',')
    scales = [float(scale) for scale in scales]
    
    # Complete calculations of the volumes of each class from the predicted masks and complete a statistical test as to whether the difference between
    # the mean volume of each class of the entered images is statistically significantly different to that of the mean healthy volume of each class

    # The results will be displayed in the terminal and saved as a CSV file in the user chose output path for access at a later date

    if args.reuse:
        if args.gm == 'healthy' or 'Healthy' or 'HEALTHY' or 'None' or 'none' or 'NONE':
            add_healthy_df_calcs(preds, classes, args.entered_hpf, scales, STAT_PATH, OUT_PATH)
        else:
            add_df_calcs(preds, classes, args.entered_hpf, args.gm, scales, STAT_PATH, OUT_PATH)

    else:
        if args.gm == 'healthy' or 'Healthy' or 'HEALTHY' or 'None' or 'none' or 'NONE':
            add_healthy_df_calcs(preds, classes, args.entered_hpf, scales, STAT_PATH, OUT_PATH)
        else:
            gm_df_calcs(preds, classes, args.entered_hpf, args.gm, scales, STAT_PATH, OUT_PATH)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_files', type=str, help='Enter filenames of the images being input (separated by space)', required=True)
    parser.add_argument('--reuse', action='store_true', help='Set to \'True\' if adding more images with a gm and hpf that have already been entered', default=False)
    parser.add_argument('--entered_hpf', type=int, help='What stage of development, in hours post-fertilisation (hpf), were the provided images taken in?', required=True)
    parser.add_argument('--gm', type=str, help='What genetic modification was applied to the zebrafish embryo prior to the image being taken?', default='Healthy')
    parser.add_argument('--scale', type=str, help='Enter the width dimension of each image in \u03bcm separated by a single space. Press ENTER when done.', required=True)
    parser.add_argument('--model_name', type=str, help='Model architecture to be used for predictions', default='AttentionUnet')
    parser.add_argument('--backbone', type=str, help='Pretrained backbone used in the model', default='vgg16')
    parser.add_argument('--epochs', type=int, help='Number of epochs the model was trained on', default=100)

    args = parser.parse_args()

    main(args)