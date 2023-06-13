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

def predict(load_path, model_name, backbone, in_paths, out_path, hpf):

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

    # Reshape array for patches for each image so that the element for each array contains an 
    # element for each patch

    imgs_reshaped = imgs.reshape(imgs.shape[0], -1, imgs.shape[4], imgs.shape[5], imgs.shape[6])

    # Use model to predict mask for each 3D patch and add the patches to an array of shape 
    # (n_images, n_patches, height, width, depth, classes)

    preds = []
    preprocess_input = get_preprocessing(backbone)

    for img_patches in imgs_reshaped:
        pred_patches = []
        for patch in img_patches:
            patch_3ch = np.stack([patch]*3, axis=-1)
            patch_3ch_add_axis = np.expand_dims(patch_3ch, axis=0).astype(np.float32)
            if backbone is not None:
                patch_3ch_input = preprocess_input(patch_3ch_add_axis)
            else:
                patch_3ch_input = patch_3ch_add_axis / 255
            patch_pred = model.predict(patch_3ch_input)
            patch_pred_argmax = np.argmax(patch_pred, axis=4)[0,:,:,:]
            pred_patches.append(patch_pred_argmax)
        pred_patches = np.asarray(pred_patches, dtype=np.ndarray)
        preds.append(pred_patches)
    preds = np.asarray(preds, dtype=np.ndarray)

    # Reshape patches to shape just after patchifying

    preds_reshaped = np.reshape(preds, imgs.shape)

    # Repatch the patches to the volume of the original images

    reconstructed_preds = []

    for pred_patches in preds_reshaped:
        reconstructed_pred = unpatchify(pred_patches, img.shape)
        reconstructed_preds.append(reconstructed_pred)
    reconstructed_preds = np.asarray(reconstructed_preds, dtype=np.ndarray)

    # Convert to uint8 for opening in image viewing software

    reconstructed_preds = reconstructed_preds.astype(np.uint8)

    # Save masks as segmented volumes

    for reconstructed_pred in reconstructed_preds:
        imsave(out_path+'{}HPF_test_pred_{}_{}.tif'.format(hpf, backbone, model_name), reconstructed_pred)

    return imgs_full_size_3ch, reconstructed_preds