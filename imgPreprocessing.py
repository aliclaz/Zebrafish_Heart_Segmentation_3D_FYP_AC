from skimage import io
import numpy as np
from patchify import patchify
from sklearn.model_selection import train_test_split

def get_hpf(hpf):
    # If the hpf entered is not equal to one of the hpfs the models were trained on, find which it is closest
    # to and use that model to train it

    abs_diff = [abs(hpf - 30), abs(hpf - 36), abs(hpf - 48)]
    all_hpf = [30, 36, 48]
    closest = min(abs_diff)
    hpf_index = abs_diff.index(closest)
    mod_hpf = all_hpf[hpf_index]

    return mod_hpf

def load_process_imgs(img_path, mask_path):

    # Load input images and masks

    image = io.imread(img_path)
    img_patches = patchify(image, (64, 64, 64), step=64)

    mask = io.imread(mask_path)
    mask_patches = patchify(mask, (64, 64, 64), step=64)

    # Reshape each array to have shape (n_patches, height, width, depth)

    imgs_reshaped = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], 
                                            img_patches.shape[5]))
    masks_reshaped = np.reshape(mask_patches, (-1, mask_patches.shape[3], mask_patches.shape[4], 
                                            mask_patches.shape[5]))
    
    # Convert image to have 3 channels, add a single channel to the masks and convert both to type np.float32

    train_imgs = np.stack((imgs_reshaped,)*3, axis=-1).astype(np.float32)
    train_masks = np.expand_dims(masks_reshaped, axis=4).astype(np.float32)
    train_masks /= 255.0

    # Split dataset into training and validation sets

    x_train, x_val, y_train, y_val = train_test_split(train_imgs, train_masks, test_size=0.1, random_state=0)

    return x_train, x_val, y_train, y_val