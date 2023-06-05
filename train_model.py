import argparse
import tensorflow as tf
from tensorflow import keras

if __name__ == '__main__':
    import os
    DEVICES = tf.config.list_physical_devices('GPU')
    print(DEVICES)
    gpu_use = [i for i in range(len(DEVICES))]
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    mult_gpus = [i + 1 for i in range(len(DEVICES) - 1)]
    if len(DEVICES) > 1:
        for i in range(len(DEVICES)):
            if i == 0:
                str_gpu_use = '0'
            else:
                str_gpu_use = str_gpu_use + ', {}'.format(mult_gpus[i])
    else:
        str_gpu_use = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str_gpu_use

import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
import pandas as pd

from imgPreprocessing import load_process_imgs
from seg_models import Unet, AttentionUnet, AttentionResUnet, get_preprocessing, losses, metrics
from display import show_history, show_all_historys, show_val_masks, show_test_masks
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
        n_imgs = 5
    elif args.hpf == 36:
        n_classes = 5
        n_imgs = 6
    elif args.hpf == 30:
        n_classes = 4
        n_imgs = 2
    else:
        n_classes = None
        n_imgs = None
    test_paths = ['{}HPF_image{}.tif'.format(args.hpf, i) for i in range(2, n_imgs + 1)]

    # Load the training masks and images into the code and preprocess both datasets

    x_train, x_val, y_train, y_val = load_process_imgs(img_path, mask_path, args.train_val_split, n_classes)

    strategy = tf.distribute.MirroredStrategy(['GPU:{}'.format(i) for i in range(len(DEVICES))])
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    batch_size = args.batch_size * strategy.num_replicas_in_sync

    with strategy.scope():

        # Define model parameters

        encoder_weights = 'imagenet'
        activation = 'softmax'
        patch_size = 64
        channels = 3

        opt = Adam(args.learning_rate)

        train_masks = np.concatenate((y_train, y_val), axis=0)
        flat_train_masks = train_masks.reshape(-1)
        class_weights = compute_class_weight('balanced', classes=np.unique(flat_train_masks), y=flat_train_masks)

        dice_loss = losses.DiceLoss(class_weights=class_weights)
        cat_focal_loss = losses.CategoricalFocalLoss()
        total_loss =  dice_loss + cat_focal_loss

        m = [metrics.IOUScore(threshold=0.5), metrics.FScore(threshold=0.5)]

        # Preprocess input data with defined backbone

        preprocess_input1 = get_preprocessing(args.backbone1)
        x_train_prep = preprocess_input1(x_train)
        x_val_prep = preprocess_input1(x_val)

        # Define model - using AttentionResUnet with a resnet34 backbone and 
        # pretrained weights

        model1 = AttentionResUnet(args.backbone1, classes=n_classes, 
                                input_shape=(patch_size, patch_size, patch_size, channels), 
                                encoder_weights=encoder_weights, activation=activation)
        model1.compile(optimizer=opt, loss=total_loss, metrics=m)

    # Summarise the model architecture

    model1.summary()

    # Define callback parameters for model1

    model_names = []
    model_name1 = 'AttentionResUnet'
    model_names.append(model_name1)

    cache_model_path = mod_path + '{}HPF_{}_{}_temp.h5'.format(args.hpf, args.backbone1, model_name1)
    best_model_path = mod_path + '{}HPF_{}_{}'.format(args.hpf, args.backbone1, model_name1) + '-{val_iou_score:.4f}-{epoch:02d}.h5'
    csv_log_path = mod_path + '{}HPF_history_{}_{}_lr_{}.csv'.format(args.hpf, args.backbone1, model_name1, args.learning_rate)

    cbs = [
        ModelCheckpoint(cache_model_path, monitor='val_loss', verbose=0),
        ModelCheckpoint(best_model_path, monitor='val_loss', verbose=0, save_best_only=True),
        ReduceLROnPlateau(monitor='val_iou_score', factor=0.95, patience=3, min_lr=1e-9, min_delta=1e-8, verbose=1, mode='max'),
        CSVLogger(csv_log_path, append=True),
        EarlyStopping(monitor='val_iou_score', patience=10, verbose=0, mode='max')
    ]

    # Train the model

    history1 = model1.fit(x_train_prep, y_train, batch_size=batch_size, epochs=args.epochs, verbose=1,
                          steps_per_epoch=args.steps_per_epoch, validation_data=(x_val_prep, y_val), callbacks=cbs)
    
    # Create lists of models, historys and backbones used

    models = historys = backbones = []

    models.append(model1)
    historys.append(history1)
    backbones.append(args.backbone1)

    # Plot the train and validation losses and IOU scores at each epoch for model 1

    show_history(history1, model_name1, args.backbone1, out_path)

    with strategy.scope():

        # Preprocess input data with defined backbone

        preprocess_input2 = get_preprocessing(args.backbone2)
        x_train_prep = preprocess_input2(x_train)
        x_val_prep = preprocess_input2(x_val)

        # Define model - using AttentionUnet with a vgg16 backbone and 
        # pretrained weights

        model2 = AttentionUnet(args.backbone2, classes=n_classes, 
                            input_shape=(patch_size, patch_size, patch_size, channels), 
                            encoder_weights=encoder_weights, activation=activation)
        model2.compile(optimizer=opt, loss=total_loss, metrics=m)

    # Summarise the model architecture

    model2.summary()

    # Define callback parameters for model2

    model_name2 = 'AttentionUnet'
    model_names.append(model_name2)

    cache_model_path = mod_path + '{}HPF_{}_{}_temp.h5'.format(args.hpf, args.backbone2, model_name2)
    best_model_path = mod_path + '{}HPF_{}_{}'.format(args.hpf, args.backbone2, model_name2) + '-{val_iou_score:.4f}-{epoch:02d}.h5'
    csv_log_path = mod_path + '{}HPF_history_{}_{}_lr_{}.csv'.format(args.hpf, args.backbone2, model_name2, args.learning_rate)

    cbs = [
        ModelCheckpoint(cache_model_path, monitor='val_loss', verbose=0),
        ModelCheckpoint(best_model_path, monitor='val_loss', verbose=10, save_best_only=True),
        ReduceLROnPlateau(monitor='val_iou_score', factor=0.95, patience=3, min_lr=1e-9, min_delta=1e-8, verbose=1, mode='max'),
        CSVLogger(csv_log_path, append=True),
        EarlyStopping(monitor='val_iou_score', patience=10, verbose=0, mode='max')
    ]


    # Train the model

    history2 = model2.fit(x_train_prep, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
                          steps_per_epoch=args.steps_per_epoch, validation_data=(x_val_prep, y_val), callbacks=cbs)
    
    # Create lists of models, historys and backbones used

    models.append(model2)
    historys.append(history2)
    backbones.append(args.backbone2)

    # Plot train and validation losses and IOU scores for model 2

    show_history(history2, model2, args.backbone2, out_path)

    with strategy.scope():

        # Preprocess input data with defined backbone

        preprocess_input2 = get_preprocessing(args.backbone2)
        x_train_prep = preprocess_input2(x_train)
        x_val_prep = preprocess_input2(x_val)

        # Define model - using Unet with a vgg16 backbone and 
        # pretrained weights

        model3 = Unet(args.backbone3, classes=n_classes, 
                                    input_shape=(patch_size, patch_size, patch_size, channels), 
                                    encoder_weights=encoder_weights, activation=activation)
        model3.compile(optimizer=opt, loss=total_loss, metrics=m)

    # Summarise the model architecture

    model3.summary()

    # Define callback parameters

    model_name3 = 'Unet'
    model_names.append(model_name3)

    cache_model_path = mod_path + '{}HPF_{}_{}_temp.h5'.format(args.hpf, args.backbone3, model_name3)
    best_model_path = mod_path + '{}HPF_{}_{}'.format(args.hpf, args.backbone3, model_name3) + '-{val_iou_score:.4f}-{epoch:02d}.h5'
    csv_log_path = mod_path + '{}HPF_history_{}_{}_lr_{}.csv'.format(args.hpf, args.backbone3, model_name3, args.learning_rate)

    cbs = [
        ModelCheckpoint(cache_model_path, monitor='val_loss', verbose=0),
        ModelCheckpoint(best_model_path, monitor='val_loss', verbose=0, save_best_only=True),
        ReduceLROnPlateau(monitor='val_iou_score', factor=0.95, patience=3, min_lr=1e-9, min_delta=1e-8, verbose=1, mode='max'),
        CSVLogger(csv_log_path, append=True),
        EarlyStopping(monitor='val_iou_score', patience=10, verbose=0, mode='max')
    ]

    # Train the model

    history3 = model3.fit(x_train_prep, y_train, batch_size=8, epochs=100, verbose=1,
                          steps_per_epoch=args.steps_per_epoch, validation_data=(x_val_prep, y_val), callbacks=cbs)
    
    # Create lists of models, git historys and backbones used

    models.append(model3)
    historys.append(history3)
    backbones.append(args.backbone3)

    # Plot the train and validation losses and IOU scores at each epoch for model 3

    show_history(history3, model_name3, args.backbone3, out_path)

    # Save model

    model3.save(mod_path+'{}HPF_{}_{}_{}epochs.h5'.format(args.hpf, args.backbone3, model_name3, args.epochs))

    # Display the historys of all models together for comparison

    show_all_historys(historys, model_names, backbones, out_path)  

    # Use each model to predict masks for each validation image

    val_preds_each_model = []

    for i in range(len(models)):
        val_preds_each_model.append(val_predict(models[i], x_val, 64))
    val_preds_each_model = np.array(val_preds_each_model)

    # Display the validation images, their actual masks and their masks predicted by each model at 3 slices

    show_val_masks(model_names, x_val, y_val, val_preds_each_model, out_path)

    # Use each model to predict masks for each test image

    test_preds_each_model = []

    for i in range(len(models)):
        test_imgs, test_preds = test_predict(models[i], backbones[i], test_paths, out_path)
        test_preds_each_model.append(test_preds)

    # Display test images, actual masks and predicted masks from each model

    show_test_masks(model_names, test_imgs, test_preds_each_model, out_path)

    # Define the class labels for each stage of development

    if args.hpf == 48:
        classes = ['Background', 'Noise', 'Endocardium', 'Atrium', 'AVC', 'Ventricle']
        train_masks = np.expand_dims(train_masks, axis=4)
        healthy_masks = np.concatenate((train_masks, test_imgs), axis=0)
        healthy_scales = [295.53, 233.31, 233.31, 246.27, 246.27]
    elif args.hpf == 36:
        classes = ['Background', 'Endocardium', 'Noise', 'Atrium', 'Ventricle']
        train_masks = np.expand_dims(train_masks, axis=4)
        healthy_masks = np.concatenate((train_masks, test_imgs), axis=0)
        healthy_scales = [221.65, 221.65, 221.65, 221.65, 221.65, 221.65]
    elif args.hpf == 30:
        classes = ['Background','Endocardium', 'Linear Heart Tube', 'Noise']
        train_masks = np.expand_dims(train_masks, axis=4)
        healthy_masks = np.concatenate((train_masks, test_imgs), axis=0)
        healthy_scales = [221.65, 221.65]

    # Calculate the means, standard deviations and confidence intervals of the volume of each class
    # Put these into a dataframe, display it and then save it as a CSV for access from the main program

    healthy_df_calcs(healthy_masks, classes, healthy_scales, args.hpf, stats_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--hpf', type=int, help='stage of development fish at in train images', required=True)
    parser.add_argument('--train_val_split', type=float, help='determines size of validation set')
    parser.add_argument('--learning_rate', type=float, help='learning rate used in training of models', required=True)
    parser.add_argument('--batch_size', type=int, help='size of the batch used to train the model during an epoch', required=True)
    parser.add_argument('--steps_per_epoch', type=int, help='number of times per epoch the batch is trained on', required=True)
    parser.add_argument('--epochs', type=int, help='number of epochs used in training', required=True)
    parser.add_argument('--backbone1', type=str, help='pretrained backbone for AttentionResUnet model', required=True)
    parser.add_argument('--backbone2', type=str, help='pretrained backbone for AttentionUnet model', required=True)
    parser.add_argument('--backbone3', type=str, help='pretrained backbone for Unet model', required=True)

    args = parser.parse_args()

    main(args)