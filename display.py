import matplotlib
#matplotlib.use('TkAgg')
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
    ax[0].title('Training and Validation Loss for {} with {} backbone'.format(model_name, backbone))
    ax[0].xlabel('Epochs')
    ax[0].ylabel('Loss')
    ax[0].legend()

    acc = history.history['iou_score']
    val_acc = history.history['val_iou_score']
    ax[1].plot(epochs, acc, 'y', label='Training IOU')
    ax[1].plot(epochs, val_acc, 'r', label='Validation IOU ')
    ax[1].title('Training and Validation IOU for {} with {} backbone'.format(model_name, backbone))
    ax[1].xlabel('Epochs')
    ax[1].ylabel('IOU')
    ax[1].legend()

    plt.savefig(out_path+'{}_history_plts.jpg')
    plt.show()

def show_all_historys(historys, model_names, backbones, out_path):

    # Plot training and validation loss and accuracy at each epoch for each model
    # historys, models and backbones are lists of the history, model name and backbone name of each model

    fig, ax = plt.subplots(len(historys), 2, figsize=(10, 5*len(historys)))

    for i in range(len(historys)):
        loss = historys[i].history['loss']
        val_loss = historys[i].history['val_loss']
        epochs = range(1, len(loss) + 1)
        ax[i,0].plot(epochs, loss, 'y', label='Training Loss')
        ax[i,0].plot(epochs, val_loss, 'r', label='Validation Loss')
        ax[i,0].title('Training and Validation Loss for {} with {} backbone'.format(model_names[i], backbones[i]))
        ax[i,0].xlabel('Epochs')
        ax[i,0].ylabel('Loss')
        ax[i,0].legend()

        acc = historys[i].history['iou_score']
        val_acc = historys[i].history['val_iou_score']
        ax[i,1].plot(epochs, acc, 'y', label='Training IOU')
        ax[i,1].plot(epochs, val_acc, 'r', label='Validation IOU ')
        ax[i,1].title('Training and Validation IOU for {} with {} backbone'.format(model_names[i], backbones[i]))
        ax[i,1].xlabel('Epochs')
        ax[i,1].ylabel('IOU')
        ax[i,1].legend()
    plt.savefig(out_path+'all_model_history_plots.jpg')


def show_val_masks(model_names, imgs, gts, preds, out_path, classes):

    # Plot the validation images, and their actual and predicted masks for each patch from each model at 3 random slices
    # Models is an array of each model name and preds is an array containing arrays of the predicted masks from each model 
    
    slices = np.random.randint(64, size=(len(imgs), 3))
    n_rows = len(imgs) * 3
    n_cols = len(model_names) + 2
    values = np.unique(gts.ravel())

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    k = 0
    for i in range(n_rows):
        if i % 3 == 0 and k != 0:
            k += 1
        for j in range(n_cols):
            if j == 0:
                ax[i,j].set_title('Validation Image')
                ax[i,j].imshow(imgs[slices[k],:,:,:,0])
            elif j == 1:
                ax[i,j].set_title('Ground Truth Mask')
                ax[i,j].imshow(gts[slices[k],:,:,:,0], cmap='gray')
                c = [ax[i,j].cmap(ax[i,j].norm(value)) for value in values]
                patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
                ax[i,j].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            else:
                ax[i,j].set_title('Predicted Mask by {}'.format(model_names[j - 2]))
                ax[i,j].imshow(preds[j-2,slices[k],:,:,:,0], cmap='gray')
                c = [ax[i,j].cmap(ax[i,j].norm(value)) for value in values]
                patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
                ax[i,j].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(out_path+'val_imgs_and_masks.jpg')
    plt.show()

def show_test_masks(model_names, imgs, preds, out_path, classes):

    # Plot images from the test set and their predicted masks from each model at 3 random slices
    # Models is an array of each model name and preds is an array containing arrays of the predicted masks from each model 

    slices = np.random.randint(256, size=(len(imgs), 3))
    n_rows = len(imgs) * 3
    n_cols = len(model_names) + 2
    values = np.unique(preds.ravel())

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    k = 0
    for i in range(n_rows):
        if i % 3 == 0 and k != 0:
            k += 1
        for j in range(n_cols):
            if j == 0:
                ax[i,j].set_title('Test Image')
                ax[i,j].imshow(imgs[slices[k],:,:,:,0])
                c = [ax[i,j].cmap(ax[i,j].norm(value)) for value in values]
                patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
                ax[i,j].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            else:
                ax[i,j].set_title('Predicted Mask by {}'.format(model_names[j - 1]))
                ax[i,j].imshow(preds[j-1,slices[k],:,:,:,0], cmap='gray')
                c = [ax[i,j].cmap(ax[i,j].norm(value)) for value in values]
                patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
                ax[i,j].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(out_path+'test_imgs_and_masks.jpg')
    plt.show()

def show_pred_masks(imgs, preds, out_path, classes):
    
    # Plot images from the test set and their predicted masks at 3 random slices
    # Here preds is an array of predicted masks for each image, only 1 model used

    slices = np
    slices = np.random.randint(256, size=(len(imgs), 3))
    n_rows = len(imgs) * 3
    values = np.unique(preds.ravel())

    fig, ax = plt.subplots(n_rows, 2, figsize=(8, 4*n_rows))
    k = 0
    for i in range(n_rows):
        if i % 3 == 0 and k != 0:
            k += 1
        for j in range(2):
            if j == 0:
                ax[i,j].set_title('Test Image')
                ax[i,j].imshow(imgs[slices[k],:,:,:,0])
            else:
                ax[i,j].set_title('Predicted Mask')
                ax[i,j].imshow(preds[slices[k],:,:,:,0], cmap='gray')
                c = [ax[i,j].cmap(ax[i,j].norm(value)) for value in values]
                patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
                ax[i,j].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(out_path+'pred_imgs_and_masks.jpg')
    plt.show()

def disp_3D_val(val_imgs, val_masks, all_val_preds, model_names, classes, out_path):
    # Plot the validation actual masks and their predicted masks for each patch in the validation set, predictions from all models

    fig = plt.figure(figsize=(8, 4*len(val_masks)))
    fig.patch.set_facecolor('darkblue')
    cube_size = 1

    for i in range(len(val_masks)):
        ax = fig.add_subplot(len(val_masks), 4, (4*i)+1, projection='3d')
        ax.set_title('Actual Mask')
        val_mask = val_masks[i].reshape(val_masks[i].shape[0], val_masks[i].shape[1], val_masks[i].shape[2])
        y, x, z = np.where(val_mask != 0)
        colours = val_mask[x, y, z]
        colours_normalized = (colours - np.min(colours)) / (np.max(colours) - np.min(colours))
        greyscale_colours = np.stack([colours_normalized]*3, axis=-1)
        ax.scatter(x, y, z, c=greyscale_colours, marker='s', s=cube_size**2)
        c = np.unique(greyscale_colours.ravel())
        patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        for j in all_val_preds:
            ax = fig.add_subplot(len(val_masks), 4, (4*i)+j+2, projection='3d')
            ax.set_title('Predicted Mask by {}'.format(model_names[j]))
            val_pred = all_val_preds[j,i].reshape(all_val_preds[j,i].shape[0], all_val_preds[j,i].shape[1], all_val_preds[j,i].shape[2])
            y, x, z = np.where(val_pred != 0)
            colours = val_pred[y, x, z]
            colours_normalized = (colours - np.min(colours)) / (np.max(colours) - np.min(colours))
            greyscale_colours = np.stack([colours_normalized]*3, axis=-1)
            ax.scatter(x, y, z, c=greyscale_colours, markers='s', s=cube_size**2)
            c = np.unique(greyscale_colours.ravel())
            patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
            ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(out_path+'val_imgs_and_masks.jpg')
    plt.show()

def disp_3D_test(test_masks, test_preds, model_names, out_path, classes):
    # Plot the predicted masks for each patch in the validation set, predictions from all models

    fig = plt.figure(figsize=(8, 4*len(test_masks)))
    fig.patch.set_facecolor('darkblue')
    cube_size = 1

    for i in range(len(test_masks)):
        ax = fig.add_subplot(len(test_masks), 4, (4*i)+1, projection='3d')
        ax.set_title('Actual Mask')
        test_mask = test_masks[i].reshape(test_masks[i].shape[0], test_masks[i].shape[1], test_masks[i].shape[2])
        y, x, z = np.where(test_mask != 0)
        colours = test_mask[x, y, z]
        colours_normalized = (colours - np.min(colours)) / (np.max(colours) - np.min(colours))
        greyscale_colours = np.stack([colours_normalized]*3, axis=-1)
        ax.scatter(x, y, z, c=greyscale_colours, markers='s', s=cube_size**2)
        c = np.unique(greyscale_colours.ravel())
        patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        ax = fig.add_subplot(len(test_masks), 4, (4*i)+2, projection='3d')
        ax.set_title('Predicted Mask by')
        test_pred = test_preds[i].reshape(test_preds[i].shape[0], test_preds[i].shape[1], test_preds[i].shape[2])
        y, x, z = np.where(test_pred != 0)
        colours = test_pred[x, y, z]
        colours_normalized = (colours - np.min(colours)) / (np.max(colours) - np.min(colours))
        greyscale_colours = np.stack([colours_normalized]*3, axis=-1)
        ax.scatter(x, y, z, c=greyscale_colours, markers='s', s=cube_size**2)
        c = np.unique(greyscale_colours.ravel())
        patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(out_path+'test_imgs_and_masks.jpg')
    plt.show()
