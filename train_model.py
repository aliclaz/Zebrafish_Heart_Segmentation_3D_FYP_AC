import argparse
import tensorflow as tf
from tensorflow import keras

if __name__ == '__main__':
    import os
    os.environ['KERAS_BACKEND'] = 'tensorflow'

import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import pandas as pd

from imgPreprocessing import load_process_imgs
from seg_models import Unet, AttentionUnet, AttentionResUnet, defAttentionResUnet, defAttentionUnet, defUnet, get_preprocessing
from seg_models import losses as l
from seg_models import metrics as m
from display import show_history, show_val_masks, show_pred_masks, disp_3D_val, disp_3D_pred
from predict_module import val_predict, test_predict
from statistical_analysis.df_manipulation import healthy_df_calcs

def main(args):
    
    # Define paths for dateset and the number of classes in the dataset

    path = os.getcwd()
    img_path = '{}HPF_image.tif'.format(args.hpf)
    mask_path = '{}HPF_mask.tif'.format(args.hpf)
    out_path = path + '/Results/'
    mod_path = path + '/Models/'
    stats_path = path + '/Stats/'
    if args.hpf == 48:
        n_classes = 6
        classes = ['Background', 'AVC', 'Endocardium' 'Noise', 'Atrium', 'Ventricle']
        n_imgs = 5
    elif args.hpf == 36:
        n_classes = 5
        classes = ['Background', 'Endocardium', 'Atrium', 'Noise', 'Ventricle']
        n_imgs = 6
    elif args.hpf == 30:
        n_classes = 4
        classes = ['Background','Endocardium', 'Linear Heart Tube', 'Noise']
        n_imgs = 2
    else:
        n_classes = None
        classes = None
        n_imgs = None
    test_paths = ['{}HPF_image{}.tif'.format(args.hpf, i) for i in range(2, n_imgs + 1)]

    # Load the training masks and images into the code and preprocess both datasets

    x_train, x_val, y_train, y_val = load_process_imgs(img_path, mask_path, args.train_val_split, n_classes)

    # Initialising mirrored distribution for multi-gpu support and adjust batch size accordingly

    devices = tf.config.list_physical_devices('GPU')

    for i in range(len(devices)):
        tf.config.experimental.set_memory_growth(devices[i], True)

    strategy = tf.distribute.MirroredStrategy(['GPU:{}'.format(i) for i in range(len(devices))])
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    batch_size = args.batch_size * strategy.num_replicas_in_sync

    print('Total batch size: ', batch_size)

    print('Batch size per device: ', args.batch_size)

    with strategy.scope():

        # Define model parameters

        encoder_weights = 'imagenet'
        activation = 'softmax'
        patch_size = 64
        channels = 3

        opt = Adam(args.learning_rate)

        dice_loss = l.DiceLoss()
        cat_focal_loss = l.CategoricalFocalLoss()
        total_loss = dice_loss + cat_focal_loss

        metrics = [m.IOUScore(threshold=0.5), m.FScore(threshold=0.5)]

        if args.backbone is not None:
            # Preprocess input data with defined backbone if using pre-trained model

            preprocess_input = get_preprocessing(args.backbone)
            x_train_prep = preprocess_input(x_train)
            x_val_prep = preprocess_input(x_val)
        
        else:
            x_train_prep = x_train / 255
            x_val_prep = x_val / 255

        # Define model depending on script argument

        if args.model_name == 'AttentionResUnet':
            model = AttentionResUnet(args.backbone, classes=n_classes, dropout=args.dropout,
                            input_shape=(patch_size, patch_size, patch_size, channels), 
                            encoder_weights=encoder_weights, activation=activation)
            
        elif args.model_name == 'DefaultAttentionResUnet':
            model = defAttentionResUnet(n_classes, input_shape=(patch_size, patch_size, patch_size, 3),
                                        dropout=args.dropout, use_batchnorm=True)

        elif args.model_name == 'AttentionUnet':
            model = AttentionUnet(args.backbone, classes=n_classes, dropout=args.dropout,
                            input_shape=(patch_size, patch_size, patch_size, channels), 
                            encoder_weights=encoder_weights, activation=activation)
        
        elif args.model_name == 'DefaultAttentionUnet':
            model = defAttentionUnet(n_classes, input_shape=(patch_size, patch_size, patch_size, 3),
                                     dropout=args.dropout, use_batchnorm=True)

        elif args.model_name == 'Unet':
            model = Unet(args.backbone, classes=n_classes, dropout=args.dropout,
                            input_shape=(patch_size, patch_size, patch_size, channels), 
                            encoder_weights=encoder_weights, activation=activation)
            
        elif args.model_name == 'DefaultUnet':
            model = defUnet(n_classes, input_shape=(patch_size, patch_size, patch_size, 3),
                            dropout=args.dropout, use_batchnorm=True)

    model.compile(optimizer=opt, loss=total_loss, metrics=metrics)

    # Summarise the model architecture

    model.summary()

    cbs = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=3, min_lr=1e-9, min_delta=1e-8, verbose=1, mode='min'),
        CSVLogger('history_{}_{}_lr_{}.csv'.format(args.backbone, args.model_name, args.learning_rate), append=True),
        EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    ]

    # Train the model

    history = model.fit(x_train_prep, y_train, batch_size=batch_size, epochs=args.epochs, verbose=1,
                        validation_data=(x_val_prep, y_val), callbacks=cbs)
    
    # Save the model for use for predictions

    model.save(mod_path + '{}HPF_{}_{}_{}_epochs.h5'.format(args.hpf, args.backbone, args.model_name, args.epochs))
    load_path = mod_path + '{}HPF_{}_{}_{}_epochs.h5'.format(args.hpf, args.backbone, args.model_name, args.epochs)

    # Plot the train and validation losses and IOU scores at each epoch for the model
    
    show_history(history, args.model_name, args.backbone, out_path)

    # Use model to predict masks for each validation image

    val_preds = val_predict(load_path, x_val, patch_size)

    # Convert train and validation masks back from categorical

    train_masks = np.argmax(y_train, axis=4)

    val_masks = np.argmax(y_val, axis=4)

    # Display validation images, their actual masks and their predicted masks by the model in 2D slices

    show_val_masks(args.model_name, args.backbone, x_val, val_masks, val_preds, out_path, classes)

    # Display the the actual masks and predicted masks in 3D

    disp_3D_val(val_masks, val_preds, args.model_name, args.backbone, classes, out_path)

    # Use model to predict masks for each validation image

    test_imgs, test_preds = test_predict(load_path, args.backbone, test_paths, out_path, args.hpf)

    # Display test images and their predicted masks from the model in 2D slices

    show_pred_masks(args.model_name, args.backbone, test_imgs, test_preds, out_path, classes)

    # Display predicted masks from test images in 3D

    disp_3D_pred(test_preds, args.model_name, args.backbone, out_path, classes)

    # Collect train and validation original masks and test predictions into healthy dataset
    # with a list of their scales

    if args.hpf == 48:
        healthy_masks = np.concatenate((train_masks, val_masks, test_preds), axis=0)
        healthy_scales = [295.53, 233.31, 233.31, 246.27, 246.27]
    elif args.hpf == 36:
        healthy_masks = np.concatenate((train_masks, val_masks, test_preds), axis=0)
        healthy_scales = [221.65, 221.65, 221.65, 221.65, 221.65, 221.65]
    elif args.hpf == 30:
        healthy_masks = np.concatenate((train_masks, val_masks, test_preds), axis=0)
        healthy_scales = [221.65, 221.65]

    # Calculate the means, standard deviations and confidence intervals of the volume of each class
    # Put these into a dataframe, display it and then save it as a CSV for access from the main program

    healthy_df_calcs(healthy_masks, classes, healthy_scales, args.hpf, stats_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--hpf', type=int, help='stage of development fish at in train images', required=True)
    parser.add_argument('--train_val_split', type=float, help='determines size of validation set')
    parser.add_argument('--model_name', type=str, help='model to be trained', required=True)
    parser.add_argument('--backbone', type=str, help='pretrained backbone for model, None if no backbone', required=True)
    parser.add_argument('--learning_rate', type=float, help='learning rate used in training of models', required=True)
    parser.add_argument('--batch_size', type=int, help='size of the batch used to train the model during an epoch', required=True)
    parser.add_argument('--epochs', type=int, help='number of epochs used in training', required=True)
    parser.add_argument('--dropout', type=float, help='degree of dropout used in model', default=None)

    args = parser.parse_args()

    main(args)