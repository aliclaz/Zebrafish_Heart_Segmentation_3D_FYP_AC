import argparse
import tensorflow as tf
from tensorflow import keras

if __name__ == '__main__':
    import os
    DEVICES = tf.config.list_physical_devices('GPU')
    print(DEVICES)
    gpu_use = [i for i in range(len(DEVICES))]
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    if len(DEVICES) > 1:
        for i in range(len(DEVICES)):
            if i == 0:
                str_gpu_use = '0'
            else:
                str_gpu_use = str_gpu_use + ',{}'.format(gpu_use[i])
    else:
        str_gpu_use = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str_gpu_use

import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
import pandas as pd

from imgPreprocessing import load_process_imgs
from seg_models import Unet, AttentionUnet, AttentionResUnet, get_preprocessing, losses, metrics
from display import show_history, show_all_historys, show_val_masks, show_test_masks, disp_3D_val, disp_3D_test
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

    # Initialising mirrored distribution for multi-gpu support and adjust batch size and steps per epoch accordingly

    strategy = tf.distribute.MirroredStrategy(['GPU:{}'.format(i) for i in range(len(DEVICES))])
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    batch_size = args.batch_size * strategy.num_replicas_in_sync

    print('Batch size per device: ', batch_size / strategy.num_replicas_in_sync)

    steps_per_epoch = (len(x_train) // batch_size) / strategy.num_replicas_in_sync

    print('Number of steps per epoch: ', steps_per_epoch)

    with strategy.scope():

        # Define model parameters

        encoder_weights = 'imagenet'
        activation = 'softmax'
        patch_size = 64
        channels = 3

        opt = Adam(args.learning_rate)

        dice_loss = losses.DiceLoss()
        cat_focal_loss = losses.CategoricalFocalLoss()
        total_loss =  dice_loss + cat_focal_loss

        m = [metrics.IOUScore(threshold=0.5), metrics.FScore(threshold=0.5)]

        # Preprocess input data with defined backbone

        preprocess_input = get_preprocessing(args.backbone)
        x_train_prep = preprocess_input(x_train)
        x_val_prep = preprocess_input(x_val)

        # Define model depending on script argument

        if args.model_name == 'AttentionResUnet':
            model = model = AttentionResUnet(args.backbone, classes=n_classes, dropout=args.dropout,
                            input_shape=(patch_size, patch_size, patch_size, channels), 
                            encoder_weights=encoder_weights, activation=activation)

        elif args.model_name == 'AttentionUnet':
            model = AttentionUnet(args.backbone, classes=n_classes, dropout=args.dropout,
                            input_shape=(patch_size, patch_size, patch_size, channels), 
                            encoder_weights=encoder_weights, activation=activation)

        elif args.model_name == 'Unet':
            model = Unet(args.backbone, classes=n_classes, dropout=args.dropout,
                            input_shape=(patch_size, patch_size, patch_size, channels), 
                            encoder_weights=encoder_weights, activation=activation)

    model.compile(optimizer=opt, loss=total_loss, metrics=m)

    # Summarise the model architecture

    model.summary()

    model_names = []
    backbones = []
    model_names.append(args.model_name)
    backbones.append(args.backbone)

    # Define callback parameters for model

    cache_model_path = mod_path + '{}HPF_{}_{}_temp.h5'.format(args.hpf, args.backbone2, model_name2)
    best_model_path = mod_path + '{}HPF_{}_{}'.format(args.hpf, args.backbone2, model_name2) + '-{val_iou_score:.4f}-{epoch:02d}.h5'
    csv_log_path = mod_path + '{}HPF_history_{}_{}_lr_{}.csv'.format(args.hpf, args.backbone2, model_name2, args.learning_rate)

    cbs = [
        ModelCheckpoint(cache_model_path, monitor='val_loss', verbose=0),
        ModelCheckpoint(best_model_path, monitor='val_loss', verbose=10, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=3, min_lr=1e-9, min_delta=1e-8, verbose=1, mode='min'),
        CSVLogger(csv_log_path, append=True),
        EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    ]


    # Train the model

    history = model.fit(x_train_prep, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
                          steps_per_epoch=steps_per_epoch, validation_data=(x_val_prep, y_val), callbacks=cbs)
    
    # Plot the train and validation losses and IOU scores at each epoch for the model
    
    show_history(history, args.model_name, args.backbone, out_path)

    # Save model

    model.save(mod_path+'{}HPF_{}_{}_{}epochs.h5'.format(args.hpf, args.backbone, args.model_name, args.epochs))

    # Use model to predict masks for each validation image

    val_preds_list = []
    val_preds = val_predict(mod_path+'{}HPF_{}_{}_{}epochs.h5'.format(args.hpf, args.backbone, args.model_name, args.epochs), x_val, 64)
    val_preds_list.append(val_preds)
    val_preds = np.array(val_preds_list)

    # Convert train and validation masks back from categorical

    train_masks = []
    for i in range(len(y_train)):
        train_mask = np.argmax(y_train[i], axis=4)
        train_masks.append(train_mask)
    train_masks = np.asarray(train_masks)

    val_masks = []
    for i in range(len(y_val)):
        val_mask = np.argmax(y_val[i], axis=4)
        val_masks.append(val_mask)
    val_masks = np.asarray(val_masks)

    # Display validation images, their actual masks and their predicted masks by the model in 2D slices

    show_val_masks(model_names, x_val, val_masks, val_preds, out_path, classes)

    # Display the the actual masks and predicted masks in 3D

    disp_3D_val(val_masks, val_preds, model_names, backbones, classes, out_path)

    # Use model to predict masks for each validation image

    test_preds_list = []
    test_imgs, test_preds = test_predict(mod_path+'{}HPF_{}_{}_{}epochs.h5'.format(args.hpf, args.backbone, args.model_name, args.epochs), args.backbone, test_paths, out_path, args.hpf)
    test_preds_list.append(test_preds)
    test_preds = np.array(val_preds_list)

    # Display test images and their predicted masks from the model in 2D slices

    show_test_masks(model_names, backbones, test_imgs, test_preds, out_path, classes)

    # Display predicted masks from test images in 3D

    disp_3D_test(test_preds, model_names, backbones, out_path, classes)

    # Collect train and validation original masks and test predictions into healthy dataset
    # with a list of their scales

    if args.hpf == 48:
        healthy_masks = np.concatenate((train_masks, val_masks, test_preds[0]), axis=0)
        healthy_scales = [295.53, 233.31, 233.31, 246.27, 246.27]
    elif args.hpf == 36:
        classes = ['Background', 'Endocardium', 'Atrium', 'Noise', 'Ventricle']
        healthy_masks = np.concatenate((train_masks, val_masks, test_preds[0]), axis=0)
        healthy_scales = [221.65, 221.65, 221.65, 221.65, 221.65, 221.65]
    elif args.hpf == 30:
        classes = ['Background','Endocardium', 'Linear Heart Tube', 'Noise']
        healthy_masks = np.concatenate((train_masks, val_masks, test_preds[0]), axis=0)
        healthy_scales = [221.65, 221.65]

    # Calculate the means, standard deviations and confidence intervals of the volume of each class
    # Put these into a dataframe, display it and then save it as a CSV for access from the main program

    healthy_df_calcs(healthy_masks, classes, healthy_scales, args.hpf, stats_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--hpf', type=int, help='stage of development fish at in train images', required=True)
    parser.add_argument('--train_val_split', type=float, help='determines size of validation set')
    parser.add_argument('--model_name', type=str, help='model to be trained', required=True)
    parser.add_argument('--backbone', type=str, help='pretrained backbone for model', required=True)
    parser.add_argument('--learning_rate', type=float, help='learning rate used in training of models', required=True)
    parser.add_argument('--batch_size', type=int, help='size of the batch used to train the model during an epoch', required=True)
    parser.add_argument('--epochs', type=int, help='number of epochs used in training', required=True)
    parser.add_argument('--dropout', type=float, help='degree of dropout used in model', default=None)

    args = parser.parse_args()

    main(args)