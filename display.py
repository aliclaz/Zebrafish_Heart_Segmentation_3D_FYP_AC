from matplotlib import pyplot as plt
import numpy as np

def show_history(history, model_name, backbone, out_path):

    # Plot training and validation loss and accuracy at each epoch for certain model
    # Model and backbone are the names of the model and backbone used

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    ax[0,0].plot(epochs, loss, 'y', label='Training Loss')
    ax[0,0].plot(epochs, val_loss, 'r', label='Validation Loss')
    ax[0,0].title('Training and Validation Loss for {} with {} backbone'.format(model_name, backbone))
    ax[0,0].xlabel('Epochs')
    ax[0,0].ylabel('Loss')
    ax[0,0].legend()

    acc = history.history['iou_score']
    val_acc = history.history['val_iou_score']
    ax[0,1].plot(epochs, acc, 'y', label='Training IOU')
    ax[0,1].plot(epochs, val_acc, 'r', label='Validation IOU ')
    ax[0,1].title('Training and Validation IOU for {} with {} backbone'.format(model_name, backbone))
    ax[0,1].xlabel('Epochs')
    ax[0,1].ylabel('IOU')
    ax[0,1].legend()

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


def show_val_masks(model_names, imgs, gts, preds, out_path):

    # Plot the validation images, and their actual and predicted masks for each patch from each model at 3 random slices
    # Models is an array of each model name and preds is an array containing arrays of the predicted masks from each model 
    
    slices = np.random.randint(64, size=(len(imgs), 3))
    n_rows = len(imgs) * 3
    n_cols = len(model_names) + 2

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
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
            else:
                ax[i,j].set_title('Predicted Mask by {}'.format(model_names[j - 2]))
                ax[i,j].imshow(preds[j-2,slices[k],:,:,:,0], cmap='gray')
    plt.savefig(out_path+'val_imgs_and_masks.jpg')
    plt.show()

def show_test_masks(model_names, imgs, preds, out_path):

    # Plot images from the test set and their predicted masks from each model at 3 random slices
    # Models is an array of each model name and preds is an array containing arrays of the predicted masks from each model 

    slices = np.random.randint(256, size=(len(imgs), 3))
    n_rows = len(imgs) * 3
    n_cols = len(model_names) + 2

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    k = 0
    for i in range(n_rows):
        if i % 3 == 0 and k != 0:
            k += 1
        for j in range(n_cols):
            if j == 0:
                ax[i,j].set_title('Test Image')
                ax[i,j].imshow(imgs[slices[k],:,:,:,0])
            else:
                ax[i,j].set_title('Predicted Mask by {}'.format(model_names[j - 1]))
                ax[i,j].imshow(preds[j-1,slices[k],:,:,:,0], cmap='gray')
    plt.savefig(out_path+'test_imgs_and_masks.jpg')
    plt.show()

def show_pred_masks(imgs, preds, out_path):
    
    # Plot images from the test set and their predicted masks at 3 random slices
    # Here preds is an array of predicted masks for each image, only 1 model used

    slices = np
    slices = np.random.randint(256, size=(len(imgs), 3))
    n_rows = len(imgs) * 3

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
    plt.savefig(out_path+'pred_imgs_and_masks.jpg')
    plt.show()
