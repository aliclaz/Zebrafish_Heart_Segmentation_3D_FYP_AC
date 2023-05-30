from skimage import io
import numpy as np
from patchify import patchify
from sklearn.model_selection import train_test_split

def load_process_imgs(path):

    # Load input images and masks

    image = io.imread(path)
    img_patches = patchify(image, (64, 64, 64), step=64)

    mask = io.imread(path)
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