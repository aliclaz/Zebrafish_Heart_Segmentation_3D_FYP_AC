from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

def show_history(history, model_name, backbone, out_path, hpf):

    # Plot training and validation loss and accuracy at each epoch for certain model
    # Model and backbone are the names of the model and backbone used

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    ax[0].plot(epochs, loss, 'y', label='Training Loss')
    ax[0].plot(epochs, val_loss, 'r', label='Validation Loss')
    ax[0].set_title('Training and Validation Loss for\n{} with {} Backbone'.format(model_name, backbone))
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    acc = history.history['iou_score']
    val_acc = history.history['val_iou_score']
    ax[1].plot(epochs, acc, 'y', label='Training IOU')
    ax[1].plot(epochs, val_acc, 'r', label='Validation IOU ')
    ax[1].set_title('Training and Validation IOU for\n{} with {} Backbone'.format(model_name, backbone))
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('IOU')
    ax[1].legend()

    plt.savefig(out_path+'{}HPF_{}_{}_history_plts.jpg'.format(hpf, backbone, model_name))
    plt.show()

def show_val_masks(model_name, backbone, imgs, gts, preds, out_path, classes, hpf):

    # Plot the validation images, and their actual and predicted masks for each patch from each model at 3 random slices
    
    slices = np.random.randint(len(imgs), size=(3))
    colours = np.unique(gts.ravel())
    colours = colours[colours != 0]
    colours_normalized = colours / np.max(colours)
    c = np.stack([colours_normalized]*3, axis=-1)

    fig, ax = plt.subplots(len(imgs)*3, 3, figsize=(24, 18*len(imgs)))

    k = 0
    for i in range(len(imgs)):
        for slice in slices:
            ax[k,0].set_title('Validation Image {}, Slice {}'.format((i+1), slice))
            ax[k,0].imshow(imgs[i,:,:,slice])

            ax[k,1].set_title('Ground Truth Mask {}, Slice {}'.format((i+1), slice))
            ax[k,1].imshow(gts[i,:,:,slice], cmap='gray')
            patches = [mpatches.Patch(ec='k', fc=c[i-1] if i != 0 else 'k', label=classes[i]) for i in range(len(classes))]
            ax[k,1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            ax[k,2].set_title('Predicted Mask {}, Slice {} by\n{}with {} Backbone'.format((i+1), slice, model_name, backbone))
            ax[k,2].imshow(preds[i,:,:,slice], cmap='gray')
            patches = [mpatches.Patch(ec='k', fc=c[i-1] if i != 0 else 'k', label=classes[i]) for i in range(len(classes))]
            ax[k,2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            
            k += 1
    plt.savefig(out_path+'{}HPF_{}_{}_val_imgs_and_masks.jpg'.format(hpf, backbone, model_name))
    plt.show()

def show_pred_masks(model_name, backbone, imgs, preds, out_path, classes, hpf):

    # Plot images from the test set and their predicted masks from each model at 3 random slices 

    slices = np.random.randint(len(imgs), size=(3))
    colours = np.unique(preds.ravel())
    colours = colours[colours != 0]
    colours_normalized = colours / np.max(colours)
    c = np.stack([colours_normalized]*3, axis=-1)

    fig, ax = plt.subplots(3*len(imgs), 2, figsize=(16, 18*len(imgs)))

    k = 0
    for i in range(len(imgs)):
        for slice in slices:
            ax[k,0].set_title('Test Image {}, Slice {}'.format((i+1), slice))
            ax[k,0].imshow(imgs[i,:,:,slice])
        
            ax[k,1].set_title('Predicted Mask {}, Slice {} by\n{} with {} Backbone'.format((i+1), slice, model_name, backbone))
            ax[k,1].imshow(preds[i,:,:,slice], cmap='gray')
            patches = [mpatches.Patch(ec='k', fc=c[i-1] if i != 0 else 'k', label=classes[i]) for i in range(len(classes))]
            ax[k,1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            k += 1
    plt.savefig(out_path+'{}HPF_{}_{}_test_imgs_and_masks.jpg'.format(hpf, backbone, model_name))
    plt.show()

def disp_3D_val(val_masks, val_preds, model_name, backbone, classes, out_path, hpf):

    # Plot the validation actual masks and their predicted masks for each patch in the validation set, predictions from all models

    fig = plt.figure(figsize=(16, 6*len(val_masks)))
    cube_size = 1
    all_colours = np.unique(val_masks.ravel())
    all_colours = all_colours[all_colours != 0]
    all_colours_normalized = all_colours / np.max(all_colours)
    fc = np.stack([all_colours_normalized]*3, axis=-1)

    for i in range(len(val_masks)):
        ax = fig.add_subplot(len(val_masks), 2, (2*i)+1, projection='3d')
        ax.set_title('Actual Mask {}'.format(i+1))
        val_mask = val_masks[i].reshape(val_masks[i].shape[0], val_masks[i].shape[1], val_masks[i].shape[2])
        y, x, z = np.where(val_mask != 0)
        colours = val_mask[y, x, z]
        colours_normalized = colours / np.max(all_colours)
        greyscale_colours = np.stack([colours_normalized]*3, axis=-1)
        ax.scatter(x, y, z, c=greyscale_colours, marker='s', s=cube_size**2)
        c = np.unique(greyscale_colours.ravel())
        patches = [mpatches.Patch(ec='k', fc=fc[i], label=classes[all_colours[i]]) for i in range(len(fc))]
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1.05), loc=2, borderaxespad=0.)

        ax = fig.add_subplot(len(val_masks), 2, (2*i)+2, projection='3d')
        ax.set_title('Predicted Mask {} by\n{} with {} backbone'.format((i+1), model_name, backbone))
        val_pred = val_preds[i].reshape(val_preds[i].shape[0], val_preds[i].shape[1], val_preds[i].shape[2])
        y, x, z = np.where(val_pred != 0)
        colours = val_pred[y, x, z]
        colours_normalized = colours / np.max(all_colours)
        greyscale_colours = np.stack([colours_normalized]*3, axis=-1)
        ax.scatter(x, y, z, c=greyscale_colours, marker='s', s=cube_size**2)
        c = np.unique(greyscale_colours.ravel())
        patches = [mpatches.Patch(ec='k', fc=fc[i], label=classes[all_colours[i]]) for i in range(len(fc))]
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1.05), loc=2, borderaxespad=0.)
    plt.savefig(out_path+'{}HPF_{}_{}_val_masks_and_preds3D.jpg'.format(hpf, backbone, model_name))
    plt.show()

def disp_3D_pred(preds, model_name, backbone, out_path, classes, hpf):

    # Plot the predicted masks for each patch in the validation set, predictions from all models

    fig = plt.figure(figsize=(8, 6*len(preds)))
    cube_size = 1
    all_colours = np.unique(preds.ravel())
    all_colours = all_colours[all_colours != 0]
    all_colours_normalized = all_colours / np.max(all_colours)
    fc = np.stack([all_colours_normalized]*3, axis=-1)

    for i in range(len(preds)):
        ax = fig.add_subplot(len(preds), 1, i+1, projection='3d')
        ax.set_title('Predicted Mask {} by\n{} with {} backbone'.format((i+1), model_name, backbone))
        pred = preds[i].reshape(preds[i].shape[0], preds[i].shape[1], preds[i].shape[2])
        y, x, z = np.where(pred != 0)
        colours = pred[y, x, z]
        colours_normalized = colours / np.max(all_colours)
        greyscale_colours = np.stack([colours_normalized]*3, axis=-1)
        ax.scatter(x, y, z, c=greyscale_colours, marker='s', s=cube_size**2)
        c = np.unique(greyscale_colours.ravel())
        patches = [mpatches.Patch(ec='k', fc=fc[i], label=classes[all_colours[i]]) for i in range(len(fc))]
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1.05), loc=2, borderaxespad=0.)
    plt.savefig(out_path+'{}HPF_{}_{}_test_preds3D.jpg'.format(hpf, backbone, model_name))
    plt.show()