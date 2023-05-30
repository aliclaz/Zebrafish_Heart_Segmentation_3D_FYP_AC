# coding: utf-8

try:
    # Using tf keras
    from tensorflow.keras import backend as K
    from classification_models.tfkeras import Classifiers as Classifiers_2D # change to own function
    from classification_models_3D.tfkeras import Classifiers as Classifiers_3D # change to own function
    print('Use TF keras')

except:
    # Using keras
    from keras import backend as K
    from classification_models.keras import Classifiers as Classifiers_2D # change to own function
    from classification_models_3D.keras import Classifiers as Classifiers_3D # change to own function
    print('Use keras')

import os
import glob
import hashlib

def convert_weights(m2, m3, out_path, target_channel):
    print('Start: {}'.format(m2.name))
    for i in range(len(m2.layers)):
        layer_2D = m2.layers[i]
        layer_3D = m3.layers[i]
        print('Extract for [{}]: {} {}'.format(i, layer_2D.__class.__name__, layer_2D.name))
        print('Set for [{}]: {} {}'.format(i, layer_3D.__class__.__name__, layer_3D.name))

        if layer_2D.name != layer_3D.name:
            print('Warning: different names!')

        weights_2D = layer_2D.get_weights()
        weights_3D = layer_3D.get_weights()
        if layer_2D.__class__.__name__ == 'Conv2D' or layer_2D.__class__.__name__ == 'DepthwiseConv2D':
            print(type(weights_2D), len(weights_2D), weights_2D[0].shape, weights_3D[0].shape)
            print(layer_2D.output_shape)
            print(layer_3D.output_shape)
            weights_3D[0][...] = 0
            if target_channel == 2:
                for j in range(weights_3D[0].shape[2]):
                    weights_3D[0][:,:,j,:,:,:] = weights_2D[0] / weights_3D[0].shape[2]
            if target_channel == 1:
                for j in range(weights_3D[0].shape[1]):
                    weights_3D[0][:,j,:,:,:] = weights_2D[0] / weights_3D[0].shape[2]
            else:
                for j in range(weights_3D[0].shape[0]):
                    weights_3D[0][j, :, :, :, :] = weights_2D[0] / weights_3D[0].shape[0]

            # Bias
            if len(weights_3D) > 1:
                print(weights_3D[1].shape, weights_2D[1].shape)
                weights_3D[1] = weights_2D[1][:weights_3D[1].shape[0]]

            m3.layers[i].set_weights(weights_3D)
        elif layer_2D.__class__.__name__ == 'Sequential' and 'convnext' in layer_2D.name:
            print('Convnext', type(weights_2D), len(weights_2D), weights_2D[0].shape, weights_3D[0].shape)
            print(layer_2D.output_shape)
            print(layer_3D.output_shape)

            if 'downsampling' in layer_2D.name:
                index_w = 2
                index_b = 3
                layer_norm_0 = 0
                layer_norm_1 = 1
            else:
                index_w = 0
                index_b = 1
                layer_norm_0 = 2
                layer_norm_1 = 3

            weights_3D[index_w][...] = 0
            if target_channel == 2:
                for j in range(weights_3D[index_w].shape[2]):
                    weights_3D[index_w][:, :, j, :, :] = weights_2D[index_w] / weights_3D[index_w].shape[2]
            if target_channel == 1:
                for j in range(weights_3D[index_w].shape[1]):
                    weights_3D[index_w][:, j, :, :, :] = weights_2D[index_w] / weights_3D[index_w].shape[1]
            else:
                for j in range(weights_3D[index_w].shape[0]):
                    weights_3D[index_w][j, :, :, :, :] = weights_2D[index_w] / weights_3D[index_w].shape[0]
            
            # Bias
            if len(weights_3D) > 1:
                print(weights_3D[index_b].shape, weights_2D[index_b].shape)
                weights_3D[index_b] = weights_2D[index_b][:weights_3D[index_b].shape[0]]

            # layer norm
            weights_3D[layer_norm_0] = weights_2D[layer_norm_0]
            weights_3D[layer_norm_1] = weights_2D[layer_norm_1]

            m3.layers[i].set_weights(weights_3D)
        elif layer_2D.__class__.__name__ == 'Normalization' and i == 2:
            if len(weights_3D) == 0:
                # Effnet v2 (it's in parameters)
                pass
        else:
            m3.layers[i].set_weights(weights_2D)

    m3.save(out_path)

def convert_models():
    include_top = False
    target_channel = 0
    shape_size_3D = (64, 64, 64, 3)
    shape_size_2D = (224, 224, 3)
    list_to_check = ['vgg16', 'resnet34']