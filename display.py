import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import matplotlib
from matplotlib.backends.backend_gtk3agg import (FigureCanvasGTK3Agg as FigureCanvas)
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

def show_history(history, model_name, backbone, out_path):

    win = Gtk.Window()
    win.connect('delete-event', Gtk.main_quit)
    win.set_default_size(400, 300)
    win.set_title('Training and Validation Losses and IOU Scores for ' + model_name + ' with {} backbone'.format(backbone))

    # Plot training and validation loss and accuracy at each epoch for certain model
    # Model and backbone are the names of the model and backbone used

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    ax[0].plot(epochs, loss, 'y', label='Training Loss')
    ax[0].plot(epochs, val_loss, 'r', label='Validation Loss')
    ax[0].set_title('Training and Validation Loss for ' + model_name + ' with {} backbone'.format(backbone))
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    acc = history.history['iou_score']
    val_acc = history.history['val_iou_score']
    ax[1].plot(epochs, acc, 'y', label='Training IOU')
    ax[1].plot(epochs, val_acc, 'r', label='Validation IOU ')
    ax[1].set_title('Training and Validation IOU for ' + model_name + ' with {}'.format(backbone))
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('IOU')
    ax[1].legend()

    plt.savefig(out_path+model_name+'_history_plts.jpg')
    
    sw = Gtk.ScrolledWindow()
    win.add(sw)
    sw.set_border_width(10)

    canvas = FigureCanvas(fig)
    canvas.set_size_request(800, 600)
    sw.add(canvas)

    win.show_all()
    Gtk.main()

def show_all_historys(historys, model_names, backbones, out_path):

    win1 = Gtk.Window()
    win1.connect('delete-event', Gtk.main_quit)
    win1.set_default_size(400, 300)
    win1.set_title('Training and Validation Losses and IOU Scores for each Model')

    # Plot training and validation loss and accuracy at each epoch for each model
    # historys, models and backbones are lists of the history, model name and backbone name of each model

    fig, ax = plt.subplots(len(historys), 2, figsize=(10, 5*len(historys)))

    for i in range(len(historys)):
        loss = historys[i].history['loss']
        val_loss = historys[i].history['val_loss']
        epochs = range(1, len(loss) + 1)
        ax[i,0].plot(epochs, loss, 'y', label='Training Loss')
        ax[i,0].plot(epochs, val_loss, 'r', label='Validation Loss')
        ax[i,0].set_title('Training and Validation Loss for ' + model_names[i] + ' with {} backbone'.format(backbones[i]))
        ax[i,0].set_xlabel('Epochs')
        ax[i,0].setylabel('Loss')
        ax[i,0].legend()

        acc = historys[i].history['iou_score']
        val_acc = historys[i].history['val_iou_score']
        ax[i,1].plot(epochs, acc, 'y', label='Training IOU')
        ax[i,1].plot(epochs, val_acc, 'r', label='Validation IOU ')
        ax[i,1].set_title('Training and Validation IOU for ' + model_names[i] + ' with {} backbone'.format(backbones[i]))
        ax[i,1].set_xlabel('Epochs')
        ax[i,1].set_ylabel('IOU')
        ax[i,1].legend()
    plt.savefig(out_path+'all_model_history_plots.jpg')

    sw = Gtk.ScrolledWindow()
    win1.add(sw)
    sw.set_border_width(10)

    canvas = FigureCanvas(fig)
    canvas.set_size_request(800, 600)
    sw.add(canvas)

    win1.show_all()
    Gtk.main()

def show_val_masks(model_names, backbones, imgs, gts, preds, out_path, classes):

    win2 = Gtk.Window()
    win2.connect('delete-event', Gtk.main_quit)
    win2.set_default_size(400, 300)
    win2.set_title('Actual and Predicted Masks from Each Model for the Validation Set at Random Slices in the Volume')

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
                ax[i,j].set_title('Predicted Mask by ' + model_names[j - 2] + ' with {} backbone'.format(backbones[j - 2]))
                ax[i,j].imshow(preds[j-2,slices[k],:,:,:,0], cmap='gray')
                c = [ax[i,j].cmap(ax[i,j].norm(value)) for value in values]
                patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
                ax[i,j].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(out_path+'val_imgs_and_masks.jpg')
    
    sw = Gtk.ScrolledWindow()
    win2.add(sw)
    sw.set_border_width(10)

    canvas = FigureCanvas(fig)
    canvas.set_size_request(800, 600)
    sw.add(canvas)

    win2.show_all()
    Gtk.main()

def show_test_masks(model_names, backbones, imgs, preds, out_path, classes):

    win3 = Gtk.Window()
    win3.connect('delete-event', Gtk.main_quit)
    win3.set_default_size(400, 300)
    win3.set_title('Predicted Masks from Each Model for the Test Set at Random Slices in the Volume')

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
                ax[i,j].set_title('Predicted Mask by ' + model_names[j - 1] + ' with {} backbone'.format(backbones[j - 2]))
                ax[i,j].imshow(preds[j-1,slices[k],:,:,:,0], cmap='gray')
                c = [ax[i,j].cmap(ax[i,j].norm(value)) for value in values]
                patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
                ax[i,j].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(out_path+'test_imgs_and_masks.jpg')
    
    sw = Gtk.ScrolledWindow()
    win3.add(sw)
    sw.set_border_width(10)

    canvas = FigureCanvas(fig)
    canvas.set_size_request(800, 600)
    sw.add(canvas)

    win3.show_all()
    Gtk.main()

def show_pred_masks(imgs, preds, out_path, classes):

    win4 = Gtk.Window()
    win4.connect('delete-event', Gtk.main_quit)
    win4.set_default_size(400, 300)
    win4.set_title('Predicted Masks for the Images Entered at Random Slices in the Volume')
    
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
    
    sw = Gtk.ScrolledWindow()
    win4.add(sw)
    sw.set_border_width(10)

    canvas = FigureCanvas(fig)
    canvas.set_size_request(800, 600)
    sw.add(canvas)

    win4.show_all()
    Gtk.main()

def disp_3D_val(val_masks, all_val_preds, model_names, backbones, classes, out_path):

    win5 = Gtk.Window()
    win5.connect('delete-event', Gtk.main_quit)
    win5.set_default_size(400, 300)
    win5.set_title('Full 3D Actual and Predicted Masks from Each Model for the Validation Set')

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
            ax.set_title('Predicted Mask by ' + model_names[j] + ' with {} backbone'.format(backbones[j]))
            val_pred = all_val_preds[j,i].reshape(all_val_preds[j,i].shape[0], all_val_preds[j,i].shape[1], all_val_preds[j,i].shape[2])
            y, x, z = np.where(val_pred != 0)
            colours = val_pred[y, x, z]
            colours_normalized = (colours - np.min(colours)) / (np.max(colours) - np.min(colours))
            greyscale_colours = np.stack([colours_normalized]*3, axis=-1)
            ax.scatter(x, y, z, c=greyscale_colours, markers='s', s=cube_size**2)
            c = np.unique(greyscale_colours.ravel())
            patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
            ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(out_path+'val_imgs_and_masks3D.jpg')
    
    sw = Gtk.ScrolledWindow()
    win5.add(sw)
    sw.set_border_width(10)

    canvas = FigureCanvas(fig)
    canvas.set_size_request(800, 600)
    sw.add(canvas)

    win5.show_all()
    Gtk.main()

def disp_3D_test(test_masks, all_test_preds, model_names, backbones, out_path, classes):

    win6 = Gtk.Window()
    win6.connect('delete-event', Gtk.main_quit)
    win6.set_default_size(400, 300)
    win6.set_title('Full 3D Actual and Predicted Masks from Each Model for the Validation Set')

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

        for j in all_test_preds:
            ax = fig.add_subplot(len(test_masks), 4, (4*i)+2, projection='3d')
            ax.set_title('Predicted Mask by ' + model_names[j] + ' with {} backbone'.format(backbones[j]))
            test_pred = all_test_preds[i].reshape(all_test_preds[i].shape[0], all_test_preds[i].shape[1], all_test_preds[i].shape[2])
            y, x, z = np.where(test_pred != 0)
            colours = test_pred[x, y, z]
            colours_normalized = (colours - np.min(colours)) / (np.max(colours) - np.min(colours))
            greyscale_colours = np.stack([colours_normalized]*3, axis=-1)
            ax.scatter(x, y, z, c=greyscale_colours, markers='s', s=cube_size**2)
            c = np.unique(greyscale_colours.ravel())
            patches = [mpatches.Patch(color=c[i], label=classes[i]) for i in range(len(classes))]
            ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(out_path+'test_imgs_and_masks3D.jpg')
    
    sw = Gtk.ScrolledWindow()
    win6.add(sw)
    sw.set_border_width(10)

    canvas = FigureCanvas(fig)
    canvas.set_size_request(800, 600)
    sw.add(canvas)

    win6.show_all()
    Gtk.main()
