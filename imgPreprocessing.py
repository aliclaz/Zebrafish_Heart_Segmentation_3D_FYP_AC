from skimage.io import imread
import numpy as np
from patchify import patchify
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

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
    mask_patches = patchify(mask, (64, 64, 64), step=64)

    # Reshape each array to have shape (n_patches, height, width, depth)

    imgs_reshaped = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], 
                                            img_patches.shape[5]))
    masks_reshaped = np.reshape(mask_patches, (-1, mask_patches.shape[3], mask_patches.shape[4], 
                                            mask_patches.shape[5]))
    
    # Convert image to have 3 channels, add 1 channels to the masks and convert both to type np.float32
    
    train_imgs = np.stack((imgs_reshaped,)*3, axis=-1).astype(np.float32)
    
    # Encode labels from 0 to number of classes - 1

    labelencoder = LabelEncoder()
    n, h, w, d = masks_reshaped.shape
    masks_flat = masks_reshaped.reshape(-1,)
    encoded_masks = labelencoder.fit_transform(masks_flat)
    encoded_masks_reshaped = encoded_masks.reshape(n, h, w, d)
    train_masks = np.expand_dims(encoded_masks_reshaped, axis=4).astype(np.float32)

    train_masks_cat = to_categorical(train_masks, num_classes=n_classes)

    # Split dataset into training and validation sets

    x_train, x_val, y_train, y_val = train_test_split(train_imgs, train_masks_cat, test_size=split, random_state=0)

    return x_train, x_val, y_train, y_val

def data_generator(x_train, y_train, batch_size):
    data_gen_args = dict(width_shift_range = 0.2,
                         height_shift_range = 0.2,
                         rescale=0.1,
                         zoom_range=0.2)
    generator = ImageDataGenerator(**data_gen_args).flow(x_train, y_train, batch_size, seed=0)
    while True:
        x_batch, y_batch = generator.next()
        yield x_batch, y_batch