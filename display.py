from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

def show_history(history, model_name, backbone, out_path):

    # Plot training and validation loss and accuracy at each epoch for certain model
    # Model and backbone are the names of the model and backbone used

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    ax[0].plot(epochs, loss, 'y', label='Training Loss')
    ax[0].plot(epochs, val_loss, 'r', label='Validation Loss')
    ax[0].set_title('Training and Validation Loss for {} with {} backbone'.format(model_name, backbone))
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    acc = history.history['iou_score']
    val_acc = history.history['val_iou_score']
    ax[1].plot(epochs, acc, 'y', label='Training IOU')
    ax[1].plot(epochs, val_acc, 'r', label='Validation IOU ')
    ax[1].set_title('Training and Validation IOU for {} with {}'.format(model_name, backbone))
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('IOU')
    ax[1].legend()

    plt.savefig(out_path+'{}_{}_history_plts.jpg'.format(backbone, model_name))
    plt.show()

def show_val_masks(model_name, backbone, imgs, gts, preds, out_path, classes):

    # Plot the validation images, and their actual and predicted masks for each patch from each model at 3 random slices
    
    slices = np.random.randint(len(imgs), size=(3))
    colours = np.unique(gts.ravel())
    colours_normalized = (colours - np.min(colours)) / (np.max(colours) - np.min(colours))
    c = np.stack([colours_normalized]*3, axis=-1)

    fig, ax = plt.subplots(len(imgs)*3, 3, figsize=(15, 12*len(imgs)))

    print(len(imgs))

    k = 0
    for i in range(3*len(imgs)):
        if i % 2 == 0 and i != 0:
            k = 0
        for j in range(3):
            if j == 0:
                ax[i,j].set_title('Validation Image')
                ax[i,j].imshow(imgs[i,:,:,slices[k]])
            elif j == 1:
                ax[i,j].set_title('Ground Truth Mask')
                ax[i,j].imshow(gts[i,:,:,slices[k]], cmap='gray')
                patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
                ax[i,j].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            else:
                ax[i,j].set_title('Predicted Mask by {} with {} backbone'.format(model_name, backbone))
                ax[i,j].imshow(preds[i,:,:,slices[k]], cmap='gray')
                patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
                ax[i,j].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        k += 1
    plt.savefig(out_path+'val_imgs_and_masks.jpg')
    plt.show()

def show_pred_masks(model_name, backbone, imgs, preds, out_path, classes):

    # Plot images from the test set and their predicted masks from each model at 3 random slices 

    slices = np.random.randint(len(imgs), size=(3))
    colours = np.unique(preds.ravel())
    colours_normalized = (colours - np.min(colours)) / (np.max(colours) - np.min(colours))
    c = np.stack([colours_normalized]*3, axis=-1)

    fig, ax = plt.subplots(3*len(imgs), 2, figsize=(10, 12*len(imgs)))

    k = 0
    for i in range(3*len(imgs)):
        if i % 2 == 0 and i != 0:
            k = 0
        ax[i,0].set_title('Test Image')
        ax[i,0].imshow(imgs[i,:,:,slices[k],0])
    
        ax[i,1].set_title('Predicted Mask by {} with {} backbone'.format(model_name, backbone))
        ax[i,1].imshow(preds[i,:,:,slices[k],0], cmap='gray')
        patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
        ax[i,1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        k += 1
    plt.savefig(out_path+'test_imgs_and_masks.jpg')
    plt.show()

def disp_3D_val(val_masks, val_preds, model_name, backbone, classes, out_path):

    # Plot the validation actual masks and their predicted masks for each patch in the validation set, predictions from all models

    fig = plt.figure(figsize=(8, 4*len(val_masks)))
    fig.patch.set_facecolor('darkblue')
    cube_size = 1

    for i in range(len(val_masks)):
        ax = fig.add_subplot(len(val_masks), 2, (2*i)+1, projection='3d')
        ax.set_title('Actual Mask')
        val_mask = val_masks[i].reshape(val_masks[i].shape[0], val_masks[i].shape[1], val_masks[i].shape[2])
        y, x, z = np.where(val_mask != 0)
        colours = val_mask[x, y, z]
        colours_normalized = (colours - np.min(colours)) / ((np.max(colours)) - np.min(colours))
        greyscale_colours = np.stack([colours_normalized]*3, axis=-1)
        ax.scatter(x, y, z, c=greyscale_colours, marker='s', s=cube_size**2)
        c = np.unique(greyscale_colours.ravel())
        patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        ax = fig.add_subplot(len(val_masks), 2, (2*i)+2, projection='3d')
        ax.set_title('Predicted Mask by {} with {} backbone'.format(model_name, backbone))
        val_pred = val_preds[i].reshape(val_preds[i].shape[0], val_preds[i].shape[1], val_preds[i].shape[2])
        y, x, z = np.where(val_pred != 0)
        colours = val_pred[y, x, z]
        colours_normalized = (colours - np.min(colours)) / (np.max(colours) - np.min(colours))
        greyscale_colours = np.stack([colours_normalized]*3, axis=-1)
        ax.scatter(x, y, z, c=greyscale_colours, markers='s', s=cube_size**2)
        c = np.unique(greyscale_colours.ravel())
        patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(out_path+'val_masks_and_preds3D.jpg')
    plt.show()

def disp_3D_pred(preds, model_name, backbone, out_path, classes):

    # Plot the predicted masks for each patch in the validation set, predictions from all models

    fig = plt.figure(figsize=(4, 4*len(preds)))
    fig.patch.set_facecolor('darkblue')
    cube_size = 1

    for i in range(len(preds)):
        ax = fig.add_subplot(len(preds), 1, i+1, projection='3d')
        ax.set_title('Predicted Mask by {} with {} backbone'.format(model_name, backbone))
        pred = preds[i].reshape(preds[i].shape[0], preds[i].shape[1], preds[i].shape[2])
        y, x, z = np.where(pred != 0)
        colours = pred[y, x, z]
        colours_normalized = (colours - np.min(colours)) / (np.max(colours) - np.min(colours))
        greyscale_colours = np.stack([colours_normalized]*3, axis=-1)
        ax.scatter(x, y, z, c=greyscale_colours, markers='s', s=cube_size**2)
        c = np.unique(greyscale_colours.ravel())
        patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(out_path+'test_preds3D.jpg')
    plt.show()