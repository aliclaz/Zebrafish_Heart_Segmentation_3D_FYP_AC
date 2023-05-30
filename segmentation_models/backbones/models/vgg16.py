import os
from .. import get_submodules_from_kwargs
from keras_applications import imagenet_utils
from ..weights import load_model_weights

preprocess_input = imagenet_utils.preprocess_input

def VGG16(weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, stride_size=2, 
          init_filters=64, max_filters=512, repetitions=(2, 2, 3, 3, 3), **kwargs):
    
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if input_tensor is None:
        input_img = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            input_img = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            input_img = input_tensor

    x = input_img

    for stage, rep in enumerate(repetitions):
        for i in range(rep):
            x = layers.Conv3D(init_filters, (3, 3, 3), activation='relu', padding='same', 
                              name='block{}_conv{}'.format(stage + 1, i + 1))(x)
            
        x = layers.MaxPooling3D(stride_size[stage], strides=stride_size[stage], name='block{}_pool'.format(stage + 1))

        init_filters *= 2
        if init_filters > max_filters:
            init_filters = max_filters

    if pooling == 'avg':
        x = layers.GlobalAverageMaxPooling3D()(x)
    elif pooling == 'max':
        x = layers.GlovalMaxPooling3D()(x)

    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = input_img

    model = models.Model(inputs, x, name='vgg16')

    if weights:
        if type(weights) == str and os.path.exists(weights):
            model.load_weights(weights)
        else:
            load_model_weights(model, 'vgg16', weights, classes, **kwargs)

    return model
