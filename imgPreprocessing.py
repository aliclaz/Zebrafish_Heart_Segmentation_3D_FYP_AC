from skimage.io import imread
import numpy as np
from patchify import patchify
from tensorflow.keras.utils import to_categorical
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

def load_process_imgs(img_path, mask_path, split, n_classes):

    # Load input images and masks

    image = imread(img_path)
    img_patches = patchify(image, (64, 64, 64), step=64)

    mask = imread(mask_path)
    mask = mask.reshape(256, 256, 256, n_classes)
    mask_channels = [[] for i in range(n_classes)]

    for i in range(256):
        for j in range(n_classes):
            temp_mask = mask[:,:,i,j]
            mask_channels[j].append(temp_mask)

    mask_channels_patches = []
    for i in range(n_classes):
        mask_channels[i] = np.array(mask_channels[i])
        mask_channel_patches = patchify(mask_channels[i], (64, 64, 64), step=64)
        mask_channels_patches.append(mask_channel_patches)
    mask_patches = np.stack(mask_channels_patches, axis=-1)

    # Reshape each array to have shape (n_patches, height, width, depth)

    imgs_reshaped = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], 
                                            img_patches.shape[5]))
    masks_reshaped = np.reshape(mask_patches, (-1, mask_patches.shape[3], mask_patches.shape[4], 
                                            mask_patches.shape[5], mask_patches.shape[6]))
    
    # Convert image to have 3 channels, add a single channel to the masks and convert both to type np.float32
    
    train_imgs = np.stack((imgs_reshaped,)*3, axis=-1).astype(np.float32)
    train_masks = masks_reshaped.astype(np.float32)
    print(train_masks.shape)

    train_class_list = []
    train_masks_list = []
    for i in range(len(train_masks)):
        for i in range(n_classes):
            train_class = train_masks[i,:,:,:,j]
            y, x, z = np.where(train_class != 0)
            train_class = (j + 1)*(train_class[y, x, z] / 6)
            train_class_list.append(train_class)
        train_mask = np.array(train_class_list)
        train_masks_list.append(train_mask)
    train_masks = np.array(train_masks_list)
    train_masks /= 255.0

    # Split dataset into training and validation sets

    x_train, x_val, y_train, y_val = train_test_split(train_imgs, train_masks, test_size=split, random_state=0)

    return x_train, x_val, y_train, y_val