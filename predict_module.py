import numpy as np
import os
from skimage import io
from patchify import patchify, unpatchify
from pretrained_seg_models import get_preprocessing
from tifffile import imsave

def val_predict(model, imgs, patch_size):
    
    """ 
    Predictions of the masks by the entered model for each image in the validation set of shape 
    (patch_size x patch_size x patch_size) 
    
    """

    val_preds = []

    for img in imgs:
        val_img = np.expand_dims(img, axis=0)
        pred = model.predict(val_img)
        pred = (pred > 0.5).astype(np.uint8)
        pred = np.argmax(pred, axis=4)
        val_preds.append(pred)
    val_preds = np.array(val_preds)
    val_preds = val_preds.reshape(val_preds.shape[0], patch_size, patch_size, patch_size, val_preds.shape[1])
    val_preds = val_preds.astype(np.uint8)

    return val_preds

def test_predict(model, backbone, in_paths, out_path):

    """ 
    Loading of images and preprocessing followed by predictions of the masks by the entered model for each image
    
    """
    # Read each image in the directory, convert it into patches and add the patches to an array

    imgs = imgs_full_size = []

    for in_path in in_paths:
        img = io.imread(in_path)
        imgs_full_size.append(img)
        patches = patchify(img, (64, 64, 64), step=64)
        imgs.append(patches)
    imgs_full_size = np.array(imgs_full_size)
    imgs = np.array(imgs)

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
                    single_patch_3ch_input = preprocess_input(np.expand_dims(single_patch_3ch, axis=0))
                    single_patch_pred = model.predict(single_patch_3ch_input)
                    single_patch_pred_argmax = np.argmax(single_patch_pred,
                                                        axis=4)[0,:,:,:]
                    pred_patches.append(single_patch_pred_argmax)
        preds.append(pred_patches)
    for i in range(len(preds)):
        preds[i] = np.array(preds[i])
    preds = np.array(preds)

    # Reshape patches to shape just after patchifying

    preds_reshaped = np.reshape(preds, (imgs.shape[0], imgs.shape[1],
                                imgs.shape[2], imgs.shape[3],
                                imgs.shape[4], imgs.shape[5],
                                imgs.shape[6]))

    # Repatch the patches to the volume of the original images

    reconstructed_imgs = []

    for i in range(len(preds_reshaped)):
        reconstructed_img = unpatchify(preds_reshaped[i], test_imgs.shape)
        reconstructed_imgs.append(reconstructed_img)
    reconstructed_imgs = np.array(reconstructed_imgs)

    # Convert to uint8 for opening in image viewing software

    recounstructed_imgs = reconstructed_imgs.astype(np.uint8)

    # Save masks as segmented volumes

    for i in reconstructed_imgs:
        imsave(out_path+'{}_mask.tif'.format(img_files[i]), reconstructed_imgs[i])

    return imgs_full_size_3ch, reconstructed_imgs
