import argparse
# coding: utf-8

if __name__ == '__main__':
    import os

    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_use)

from pathlib import Path
import os
import numpy as np
from keras.optimizers import Adam
from sklearn.utils import compute_class_weight
import pandas as pd

from imgPreprocessing import load_process_imgs
from pretrained_seg_models import Unet, AttentionUnet, AttentionResUnet, get_preprocessing, losses, metrics
from display import show_history, show_all_historys, show_val_masks, show_test_masks
from predict_module import val_predict, predict
from statistical_analysis.df_manipulation import healthy_df_calcs

def main(args):

    # Define paths for dateset and the number of classes in the dataset

    img_path = Path('48_1_image.tif').parent.absolute() + '48_1_image.tif'
    print(img_path)
    mask_path = Path('48_1_mask.tif').parent.absolute() + '48_1_mask.tif'
    test_path = Path('/Images').parent.absolute() + '/Images'
    out_path = Path('/Results').parent.absolute() + '/Results'
    mod_path = Path('/Models').parent.absolute() + '/Models'
    stats_path = Path('/Stats').parent.absolute() + '/Stats'
    if args.hpf == 48:
        img_path = 6
    elif args.hpf == 36:
        n_classes = 5
    elif args.hpf == 30:
        n_classes = 4
    else:
        n_classes = 1

    # Load the training masks and images into the code and preprocess both datasets

    x_train, x_val, y_train, y_val = load_process_imgs(img_path, mask_path)

    # Define model parameters

    encoder_weights = 'imagenet'
    backbone1 = 'resnet34'
    activation = 'softmax'
    patch_size = 64
    channels = 3

    LR = 0.0001
    opt = Adam(LR)

    train_masks = np.concatenate(y_train, y_val)
    flat_train_masks = train_masks.reshape(-1)
    class_weights = compute_class_weight('balanced', classes=np.unique(flat_train_masks), y=flat_train_masks)

    total_loss = losses.DiceLoss(class_weights=class_weights) + losses.CategoricalFocalLoss()

    metrics = [metrics.IOUScore(threshold=0.5), metrics.FScore(threshold=0.5)]

    # Define model parameters

    encoder_weights = 'imagenet'
    backbone1 = 'resnet34'
    activation = 'softmax'
    patch_size = 64
    channels = 3

    LR = 0.0001
    opt = Adam(LR)

    train_masks = np.concatenate(y_train, y_val)
    flat_train_masks = train_masks.reshape(-1)
    class_weights = compute_class_weight('balanced', classes=np.unique(flat_train_masks), y=flat_train_masks)

    total_loss = losses.DiceLoss(class_weights=class_weights) + losses.CategoricalFocalLoss()

    metrics = [metrics.IOUScore(threshold=0.5), metrics.FScore(threshold=0.5)]
    # Preprocess input data

    preprocess_input1 = get_preprocessing(backbone1)
    x_train_prep = preprocess_input1(x_train)
    x_val_prep = preprocess_input1(x_val)

    # Define model - using AttentionResUnet with a resnet34 backbone and 
    # pretrained weights

    model1 = AttentionResUnet(backbone1, classes=n_classes, 
                                input_shape=(patch_size, patch_size, patch_size, channels), 
                                encoder_weights=encoder_weights, activation=activation)
    model1.compile(optimizer=opt, loss=total_loss, metrics=metrics)
    model1.summary()

    # Train the model

    history1 = model1.fit(x_train_prep, y_train, batch_size=8, epochs=100, verbose=1,
                        validation_data=(x_val_prep, y_val))
    
    # Create a list of model names, historys and backbones used

    models = model_names = historys = backbones = []

    models.append(model1)
    model_name1 = 'AttentionResUnet'
    model_names.append(model_name1)
    historys.append(history1)
    backbones.append(backbone1)

    # Plot the train and validation losses and IOU scores at each epoch for model 1

    show_history(history1, model_name1, backbone1, out_path)
    # Save the model for use in the future

    model1.save(mod_path+'{}HPF_{}_{}_100epochs.h5'.format(args.hpf, backbone1, model_name1))

    # Preprocess input data with vgg16 backbone

    backbone2 = 'vgg16'

    preprocess_input2 = get_preprocessing(backbone2)
    x_train_prep = preprocess_input2(x_train)
    x_val_prep = preprocess_input2(x_val)

    # Define model - using AttentionUnet with a vgg16 backbone and 
    # pretrained weights

    model2 = AttentionUnet(backbone2, classes=n_classes, 
                                input_shape=(patch_size, patch_size, patch_size, channels), 
                                encoder_weights=encoder_weights, activation=activation)
    model2.compile(optimizer=opt, loss=total_loss, metrics=metrics)
    model2.summary()

    # Train the model

    history2 = model2.fit(x_train_prep, y_train, batch_size=8, epochs=100, verbose=1,
                        validation_data=(x_val_prep, y_val))
    # Create a list of model names, historys and backbones used

    models.append(model2)
    model_name2 = 'AttentionUnet'
    model_names.append(model_name2)
    historys.append(history2)
    backbones.append(backbone2)

    # Plot train and validation losses and IOU scores for model 2

    show_history(history2, model2, backbone2, out_path)

    model2.save(mod_path+'{}HPF_{}_{}_100epochs.h5'.format(args.hpf, backbone2, model_name2))

    # Define model - using Unet with a vgg16 backbone and 
    # pretrained weights

    model3 = Unet(backbone2, classes=n_classes, 
                                input_shape=(patch_size, patch_size, patch_size, channels), 
                                encoder_weights=encoder_weights, activation=activation)
    model3.compile(optimizer=opt, loss=total_loss, metrics=metrics)
    model3.summary()

    # Train the model

    history3 = model3.fit(x_train_prep, y_train, batch_size=8, epochs=100, verbose=1,
                        validation_data=(x_val_prep, y_val))
    
    # Create a list of model names, historys and backbones used

    models.append(model3)
    model_name3 = 'Unet'
    model_names.append(model_name3)
    historys.append(history3)
    backbones.append(backbone2)

    # Plot the train and validation losses and IOU scores at each epoch for model 3

    show_history(history3, model_name3, backbone2, out_path)

    # Save model

    model3.save(mod_path+'{}HPF_{}_{}_100epochs.h5'.format(args.hpf, backbone2, model_name3))

    # Display the historys of all models together for comparison

    show_all_historys(historys, model_names, backbones, out_path)

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
        test_imgs, test_preds = predict(models[i], backbones[i], test_path, out_path)
        test_preds_each_model.append(test_preds)

    # Display the validation images, their actual masks and their masks predicted by each model at 3 slices

    show_val_masks(model_names, x_val, y_val, val_preds_each_model, out_path)

    # Use each model to predict masks for each test image

    test_preds_each_model = []

    for i in range(len(models)):
        test_imgs, test_preds = predict(models[i], backbones[i], test_path, out_path)
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
        classes = ['Background', 'Noise', 'Endocardium', 'Atrium', 'Ventricle']
        train_masks = np.expand_dims(train_masks, axis=4)
        healthy_masks = np.concatenate((train_masks, test_imgs), axis=0)
        healthy_scales = [221.65, 221.65, 221.65, 221.65, 221.65, 221.65]
    elif args.hpf == 30:
        classes = ['Background', 'Noise', 'Endocardium', 'Linear Heart Tube']
        train_masks = np.expand_dims(train_masks, axis=4)
        healthy_masks = np.concatenate((train_masks, test_imgs), axis=0)
        healthy_scales = [221.65, 221.65]

    # Calculate the means, standard deviations and confidence intervals of the volume of each class
    # Put these into a dataframe, display it and then save it as a CSV for access from the main program

    healthy_df_calcs(healthy_masks, classes, healthy_scales, args.hpf, out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--hpf', type=int, help='stage of development fish at in train images', required=True)

    args = parser.parse_args()

    main(args)