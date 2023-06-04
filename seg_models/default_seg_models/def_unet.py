from keras_applications import get_submodules_from_kwargs

from ._common_blocks import Conv3DBn, Conv3DTrans, MaxPool3D

backend = None
layers = None
models = None
keras_utils = None

def get_submodules():
    return {'backend': backend, 'models': models, 'layers': layers, 'utils': keras_utils}

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv3DBn(filters, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                        use_batchnorm=use_batchnorm, name=name, **kwargs)(input_tensor)
    
    return wrapper

def encoder_block(filters, max_pooling=True, use_batchnorm=False, name=None):
    kwargs = get_submodules()

    conv1_name = name + 'a'
    conv2_name = name + 'b'

    def wrapper(input_tensor):
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(input_tensor)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)
        if max_pooling:
            out_tensor = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name=name, **kwargs)(x)
        else:
            out_tensor = x
        skip = x

        return out_tensor, skip
    
    return wrapper

def DecoderBlock(filters, stage, use_batchnorm=False):
    kwargs = get_submodules()

    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    def layer(input_tensor, skip=None):

        x = Conv3DTrans(filters, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='same', use_batchnorm=use_batchnorm, 
                        name=transp_name, **kwargs)(input_tensor)

        if skip is not None:
            x = layers.Concatenate(axis=4, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer

def defUnet(n_classes, input_shape=(None, None, None, 3), use_batchnorm=False, dropout=False):
    
    """ Define size of input layer """
    inputs = Input(input_shape)

    steps = 4
    features = 64
    drop_rate = 0.1
    skips = []

    """ Encoder """
    for i in range(steps):
        if i > 1 and dropout:
            drop_rate = 0.2
        x, y = encoder_block(x, features, drop_rate, max_pooling=True)
        skips.append(y)
        features *= 2
    x, = encoder_block(x, features, drop_rate, max_pooling=False)

    # Expansive path
    for i in reversed(range(steps)):
        features //= 2
        if i < 2:
            drop_rate = 0.1
        x = decoder_block(x, skips[i], features, drop_rate)

    # Final convolution to produce channel for each filter
    if n_classes == 1:
        activation = 'sigmoid'
    elif n_classes > 1:
        activation = 'softmax'
    outputs = Conv3D(n_classes, (1, 1, 1), activation=activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model