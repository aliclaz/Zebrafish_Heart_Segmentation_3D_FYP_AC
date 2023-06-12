import numpy as np
import os
from keras.models import load_model
from skimage import io
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from patchify import patchify, unpatchify
from imgPreprocessing import get_hpf
from seg_models import get_preprocessing
from tifffile import imsave

def val_predict(load_path, imgs, patch_size):
    
    """ 
    Predictions of the masks by the entered model for each image in the validation set of shape 
    (patch_size x patch_size x patch_size) 
    
    """
    model = load_model(load_path, compile=False)

    val_preds = []

    for img in imgs:
        val_img = np.expand_dims(img, axis=0)
        pred = model.predict(val_img)
        pred = (pred > 0.5).astype(np.uint8)
        pred = np.argmax(pred, axis=4)
        val_preds.append(pred)
    val_preds = np.asarray(val_preds, dtype=np.ndarray)
    val_preds = val_preds.reshape(val_preds.shape[0], patch_size, patch_size, patch_size, val_preds.shape[1])
    val_preds = val_preds.astype(np.uint8)

    return val_preds

def test_predict(load_path, backbone, in_paths, out_path, hpf):

    """ 
    Loading of images and preprocessing followed by predictions of the masks by the entered model for each image
    
    """

    model = load_model(load_path, compile=False)

    # Read each image in the directory, convert it into patches and add the patches to an array

    imgs = []
    imgs_full_size = []

    for in_path in in_paths:
        img = io.imread(in_path)
        imgs_full_size.append(img)
        patches = patchify(img, (64, 64, 64), step=64)
        imgs.append(patches)
    imgs_full_size = np.asarray(imgs_full_size, dtype=np.ndarray)
    imgs = np.asarray(imgs, dtype=np.ndarray)

    # Convert full sized image to have 3 channels for display purposes

    imgs_full_size_3ch = np.stack((imgs_full_size,)*3, axis=-1).astype(np.uint8)

    # Use model to predict mask for each 3D patch and add the patches to an array of shape 
    # (n_images, n_patches, height, width, depth, classes)

    pred_patches = []
    preds = []
    preprocess_input = get_preprocessing(backbone)

    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            for k in range(imgs.shape[2]):
                for l in range(imgs.shape[3]):
                    single_patch = imgs[i,j,k,l,:,:,:]
                    single_patch_3ch = np.stack((single_patch,)*3, axis=-1)
                    single_patch_3ch_size5 = np.expand_dims(single_patch_3ch, axis=0).astype(np.float32)
                    single_patch_3ch_input = preprocess_input(single_patch_3ch_size5)
                    single_patch_pred = model.predict(single_patch_3ch_input)
                    single_patch_pred_argmax = np.argmax(single_patch_pred,
                                                        axis=4)[0,:,:,:]
                    pred_patches.append(single_patch_pred_argmax)
        preds.append(pred_patches)
    for i in range(len(preds)):
        preds[i] = np.asarray(preds[i], dtype=np.ndarray)
    preds = np.asarray(preds, dtype=np.ndarray)

    # Reshape patches to shape just after patchifying

    preds_reshaped = np.reshape(preds, (imgs.shape[0], imgs.shape[1],
                                imgs.shape[2], imgs.shape[3],
                                imgs.shape[4], imgs.shape[5],
                                imgs.shape[6]))

    # Repatch the patches to the volume of the original images

    reconstructed_preds = []

    for i in range(len(preds_reshaped)):
        reconstructed_pred = unpatchify(preds_reshaped[i], imgs_full_size.shape)
        reconstructed_preds.append(reconstructed_pred)
    reconstructed_preds = np.asarray(reconstructed_preds, dtype=np.ndarray)

    # Convert to uint8 for opening in image viewing software

    reconstructed_preds = reconstructed_preds.astype(np.uint8)

    # Save masks as segmented volumes

    for i in reconstructed_preds:
        imsave(out_path+'{}HPF_test_pred.tif'.format(hpf), reconstructed_preds[i])

    return imgs_full_size_3ch, reconstructed_preds

def predict(load_path, backbone, in_paths, out_path, hpf, GM):

    """ 
    Loading of images and preprocessing followed by predictions of the masks by the entered model for each image
    
    """

    model = load_model(load_path, compile=False)

    mod_hpf = get_hpf(hpf)

    # Read each image in the directory, convert it into patches and add the patches to an array

    height = width = depth = 256
    imgs = imgs_256x256x256 = []
    lower = 1 if mod_hpf == 30 else 2

    for in_path in in_paths:
        img_full_size = io.imread(in_path)
        img_256x256x256 = resize(img_full_size, (height, width, depth), order=1, mode='mean')
        img_256x256x256 = rescale_intensity(img_256x256x256, in_range=(lower, 4))
        imgs_256x256x256.append(img_256x256x256)
        patches = patchify(img_256x256x256, (64, 64, 64), step=64)
        imgs.append(patches)
    imgs_256x256x256 = np.asarray(imgs_256x256x256, dtype=np.ndarray)
    imgs = np.asarray(imgs, dtype=np.ndarray)

    # Convert full sized image to have 3 channels for display purposes

    imgs_256x256x256_3ch = np.stack((imgs_256x256x256,)*3, axis=-1).astype(np.uint8)

    # Use model to predict mask for each 3D patch and add the patches to an array of shape 
    # (n_images, n_patches, height, width, depth, classes)

    pred_patches = []
    preds = []
    preprocess_input = get_preprocessing(backbone)

    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            for k in range(imgs.shape[2]):
                for l in range(imgs.shape[3]):
                    single_patch = imgs[i,j,k,l,:,:,:]
                    single_patch_3ch = np.stack((single_patch,)*3, axis=-1)
                    single_patch_3ch_input = preprocess_input(np.expand_dims(single_patch_3ch, axis=0))
                    single_patch_pred = model.predict(single_patch_3ch_input)
                    single_patch_pred_argmax = np.argmax(single_patch_pred,
                                                        axis=4)[0,:,:,:]
                    pred_patches.append(single_patch_pred_argmax)
        preds.append(pred_patches)
    for i in range(len(preds)):
        preds[i] = np.asarray(preds[i], dtype=np.ndarray)
    preds = np.asarray(preds, dtype=np.ndarray)

    # Reshape patches to shape just after patchifying

    preds_reshaped = np.reshape(preds, (imgs.shape[0], imgs.shape[1],
                                imgs.shape[2], imgs.shape[3],
                                imgs.shape[4], imgs.shape[5],
                                imgs.shape[6]))

    # Repatch the patches to the volume of the original images

    reconstructed_preds = []

    for i in range(len(preds_reshaped)):
        reconstructed_pred = unpatchify(preds_reshaped[i], imgs_256x256x256.shape)
        reconstructed_preds.append(reconstructed_pred)
    reconstructed_preds = np.asarray(reconstructed_preds, dtype=np.ndarray)

    # Convert to uint8 for opening in image viewing software

    reconstructed_preds = reconstructed_preds.astype(np.uint8)

    # Save masks as segmented volumes

    for i in reconstructed_preds:
        imsave(out_path+'{}HPF_{}_predicted_mask.tif'.format(hpf, GM), reconstructed_preds[i])

    return imgs_256x256x256_3ch, reconstructed_preds