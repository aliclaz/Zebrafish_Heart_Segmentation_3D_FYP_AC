import os
from .. import get_submodules_from_kwargs
from keras_applications import imagenet_utils
from ..weights import load_model_weights

preprocess_input = imagenet_utils.preprocess_input

def VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, stride_size=2, 
          init_filters=64, max_filters=512, repetitions=(2, 2, 3, 3, 3), **kwargs):
    
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')
    
    # if stride_size is scalar make it tuple of length 5 with elements tuple of size 3
    # (stride for each dimension for more flexibility)
    if type(stride_size) not in (tuple, list):
        stride_size = [
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
        ]
    else:
        stride_size = list(stride_size)

    if len(stride_size) < 3:
        print('Error: stride_size length must be 3 or more')
        return None

    if len(stride_size) != len(repetitions):
        print('Error: stride_size length must be equal to repetitions length - 1')
        return None

    for i in range(len(stride_size)):
        if type(stride_size[i]) not in (tuple, list):
            stride_size[i] = (stride_size[i], stride_size[i], stride_size[i])

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

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
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
            load_model_weights(model, 'vgg16', weights, classes, include_top, **kwargs)

    return model
