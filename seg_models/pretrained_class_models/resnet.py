import os
import collections

from .. import get_submodules_from_kwargs
from .weights import load_model_weights

backend = None
layers = None
models = None
keras_utils = None

ModelParams = collections.namedtuple('ModelParams', ['model_name', 'repetitions', 'residual_block'])

def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base +'relu'
    sc_name = name_base + 'sc'

    return conv_name, bn_name, relu_name, sc_name

def get_conv_params(**params):
    default_conv_params = {'kernel_initializer': 'he_uniform', 'use_bias': False, 'padding': 'valid'}
    default_conv_params.update(params)

    return default_conv_params

def get_bn_params(**params):
    default_bn_params = {'axis': 4, 'momentum': 0.99, 'epsilon': 2e-5, 'center': True, 'scale': True}
    default_bn_params.update(params)

    return default_bn_params

def residual_conv_block(filters, stage, block, strides=(1, 1, 1), cut='pre'):
    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = layers.BatchNormalization(name=bn_name+ '1', **bn_params)(input_tensor)
        x = layers.Activation('relu', name=relu_name + '1')(x)

        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = layers.Conv3D(filters, (1, 1, 1), name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')
        
        x = layers.ZeroPadding3D(padding=(1, 1, 1))(x)
        x = layers.Conv3D(filters, (3, 3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)

        x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)
        x = layers.ZeroPadding3D(padding=(1, 1, 1))(x)
        x = layers.Conv3D(filters, (3, 3, 3), name=conv_name + '2', **conv_params)(x)

        x = layers.add([x, shortcut])

        return x
    
    return layer

def residual_bottleneck_block(filters, stage, block, strides=None, cut='pre'):
    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = layers.Activation('relu', name=relu_name + '1')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = layers.Conv3D(filters * 4, (1, 1, 1), name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = layers.Conv3D(filters, (1, 1, 1), name=conv_name + '1', **conv_params)(x)

        x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)
        x = layers.ZeroPadding3D(padding=(1, 1, 1))(x)
        x = layers.Conv3D(filters, (3, 3, 3), strides=strides, name=conv_name + '2', **conv_params)(x)

        x = layers.BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '3')(x)
        x = layers.Conv3D(filters * 4, (1, 1, 1), name=conv_name + '3', **conv_params)(x)

        # add residual connection
        x = layers.Add()([x, shortcut])

        return x
    return layer

def ResNet(model_params, input_shape=None, input_tensor=None, include_top=False, pooling=None, classes=1000, stride_size=2, 
           init_filters=64, weights='imagenet', repetitions=None, **kwargs):
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

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

    if len(stride_size) - 1 != len(repetitions):
        print('Error: stride_size length must be equal to repetitions length - 1')
        return None

    for i in range(len(stride_size)):
        if type(stride_size[i]) not in (tuple, list):
            stride_size[i] = (stride_size[i], stride_size[i], stride_size[i])

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, name='data')
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    ResidualBlock = model_params.residual_block

    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()

    x = layers.BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
    x = layers.ZeroPadding3D(padding=(3, 3, 3))(x)
    x = layers.Conv3D(init_filters, (7, 7, 7), strides=stride_size[0], name='conv0', **conv_params)(x)
    x = layers.BatchNormalization(name='bn0', **bn_params)(x)
    x = layers.Activation('relu', name='relu0')(x)
    x = layers.ZeroPadding3D(padding=(1, 1, 1))(x)
    pool = (stride_size[1][0] + 1, stride_size[1][1] + 1, stride_size[1][2] + 1)
    x = layers.MaxPooling3D(pool, strides=stride_size[1], padding='valid', name='pooling0')(x)

    stride_count = 2
    for stage, rep in enumerate(repetitions):
        for block in range(rep):

            filters = init_filters * (2 ** stage)

            # first block of first stage without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = ResidualBlock(filters, stage, block, strides=(1, 1, 1), cut='post')(x)

            elif block == 0:
                x = ResidualBlock(filters, stage, block, strides=stride_size[stride_count], cut='post')(x)
                stride_count += 1
            else:
                x = ResidualBlock(filters, stage, block, strides=(1, 1, 1), cut='pre')(x)

    x = layers.BatchNormalization(name='bn1', **bn_params)(x)
    x = layers.Activation('relu', name='relu1')(x)

    if pooling == 'avg':
        x = layers.GlobalAveragePooling3D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling3D(name='max_pool')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x)

    if weights:
        if type(weights) == str and os.path.exists(weights):
            model.load_weights(weights)
        else:
            load_model_weights(model, model_params.model_name,
                               weights, classes, include_top, **kwargs)

    return model

MODEL_PARAMS = {
    'resnet18': ModelParams('resnet18', (2, 2, 2, 2), residual_conv_block),
    'resnet34': ModelParams('resnet34', (3, 4, 6, 3), residual_conv_block),
    'resnet50': ModelParams('resnet50', (3, 4, 6, 3), residual_bottleneck_block),
}

def ResNet18(input_shape=None, input_tensor=None, weights=None, classes=1000, stride_size=2, init_filters=64, include_top=False,
             repetitions=(2, 2, 2, 2), **kwargs):

    return ResNet(MODEL_PARAMS['resnet18'], input_shape=input_shape, input_tensor=input_tensor, weights=weights, classes=classes, 
                  stride_size=stride_size, init_filters=init_filters, include_top=include_top, repetitions=repetitions, **kwargs)

def ResNet34(input_shape=None, input_tensor=None, weights=None, classes=1000, stride_size=2, init_filters=64, include_top=False,
             repetitions=(3, 4, 6, 3), **kwargs):

    return ResNet(MODEL_PARAMS['resnet34'], input_shape=input_shape, input_tensor=input_tensor, weights=weights, classes=classes, 
                  stride_size=stride_size, init_filters=init_filters, include_top=include_top, repetitions=repetitions, **kwargs)

def ResNet50(input_shape=None, input_tensor=None, weights=None, classes=1000, stride_size=2, init_filters=64, include_top=False,
             repetitions=(3, 4, 6, 3), **kwargs):
    
    return ResNet(MODEL_PARAMS['resnet50'], input_shape=input_shape, input_tensor=input_tensor, weights=weights, classes=classes, 
                  stride_size=stride_size, init_filters=init_filters, include_top=include_top, repetitions=repetitions, **kwargs)

def preprocess_input(x, **kwargs):
    return x

setattr(ResNet18, '__doc__', ResNet.__doc__)
setattr(ResNet34, '__doc__', ResNet.__doc__)
setattr(ResNet50, '__doc__', ResNet.__doc__)