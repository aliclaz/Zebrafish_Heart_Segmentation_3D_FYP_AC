from tensorflow.keras.applications import get_submodules_from_kwargs

from ._common_blocks import Conv3DBn, Conv3DTrans, UpSamp3D, AddAct, Mult, MaxPool3D

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

def ResConvBlock(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        x = Conv3x3BnReLU(filters, use_batchnorm, name=name+'a')(input_tensor)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=name+'b')(x)
        shortcut = Conv3DBn(filters, 1, kernel_initializer='he_uniform', padding='same', use_batchnorm=use_batchnorm, name=name, **kwargs)(input_tensor)
        x = AddAct('relu', name=name, **kwargs)([shortcut, x])

        return x
    return wrapper

def RepeatElement(tensor, rep):
    return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4), arguments={'repnum': rep})(tensor)   

def GatingSignal(filters, use_batchnorm, name=None):
    kwargs = get_submodules()
    
    def wrapper(input_tensor):
        return Conv3DBn(filters, kernel_size=(1, 1, 1), activation='relu', padding='same', kernel_initializer='he_uniform', 
                        use_batchnorm=use_batchnorm, name=name, **kwargs)(input_tensor)
    
    return wrapper

def AttentionBlock(inter_shape, use_batchnorm, name=None):
    kwargs = get_submodules()

    conv1_name = name + '_theta_x'
    conv2_name = name + '_phi_g'
    conv3_name = name + '_sigmoid_xg'
    conv4_name = name + '_out'

    def wrapper(skip_connection, gating):
        shape_x = backend.int_shape(skip_connection)
        shape_g = backend.int_shape(gating)

        theta_x = Conv3DBn(inter_shape, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initalizer='he_normal',
                           use_batchnorm=use_batchnorm, name=conv1_name, **kwargs)(skip_connection)
        shape_theta_x = backend.int_shape(theta_x)

        phi_g = Conv3DBn(inter_shape, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal',
                         use_batchnorm=use_batchnorm, name=conv2_name, **kwargs)(gating)
        upsample_g = Conv3DTrans(inter_shape, (3, 3, 3), padding='same', strides=(shape_theta_x[1] // shape_g[1],
                                                                                 shape_theta_x[2] // shape_g[2],
                                                                                 shape_theta_x[3] // shape_g[3]),
                                                                                 name=name, **kwargs)(phi_g)
        
        act_xg = AddAct('relu', name=name, **kwargs)([upsample_g, theta_x])
        sigmoid_xg = Conv3DBn(1, kernel_size=(1, 1, 1), activation='softmax', kernel_initializer='he_normal', padding='same',
                              use_batchnorm=use_batchnorm, name=conv3_name, **kwargs)(act_xg)
        shape_sigmoid = backend.int_shape(sigmoid_xg)
        upsample_psi = UpSamp3D(size=(shape_x[1] // shape_sigmoid[1], 
                                       shape_x[2] // shape_sigmoid[2], 
                                       shape_x[3] // shape_sigmoid[3]), name=name, **kwargs)(sigmoid_xg)
        upsample_psi = RepeatElement(upsample_psi, shape_x[4])

        y = Mult(**kwargs, name=name)([upsample_psi, skip_connection])

        result = Conv3DBn(shape_x[4], (1, 1, 1), kernel_intitializer='he_normal', padding='same', use_batchnorm=True, 
                          name=conv4_name, **kwargs)(y)
        
        return result
    
    return wrapper

def DecoderBlock(filters, stage, use_batchnorm=False):
    kwargs = get_submodules()

    gate_name = 'decoder_stage{}_gating'.format(stage)
    atten_name = 'decoder_stage{}_attention'.format(stage)
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    res_conv_block_name = 'decoder_stage{}'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    def layer(input_tensor, skip=None):
        x = GatingSignal(filters, use_batchnorm, name=gate_name)(input_tensor)
        if skip is not None:
            atten = AttentionBlock(filters, use_batchnorm, name=atten_name)(skip, x)
        x = UpSamp3D(size=(2, 2, 2), name=up_name, **kwargs)(input_tensor)
        if skip is not None:
            x = layers.Concatenate(axis=4, name=concat_name)([x, atten])
        
        x = ResConvBlock(filters, use_batchnorm=use_batchnorm, name=res_conv_block_name)(x)

        return x

    return layer

def defAttentionResUnet(n_classes, input_shape=(None, None, None, 3), use_batchnorm=False, dropout=False, **kwargs):
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    """ Define size of input layer """
    inputs = layers.Input(input_shape)
    x = inputs

    """ Define the number of steps and the number of filters in the first encoder block """
    steps = 4
    features = 64
    skips = []

    """ Encoder """
    for i in range(steps):
        x, y = ResConvBlock(features, use_batchnorm=use_batchnorm, name='encoder_block{}'.format(i))(x)
        skips.append(y)
        features *= 2

    """ Centre block """
    x, _ = ResConvBlock(features, use_batchnorm=use_batchnorm, name='centre_block')(x)

    """ Decoder """
    for i in reversed(range(steps)):
        features //= 2
        x = DecoderBlock(features, i, use_batchnorm=use_batchnorm)(x, skips[i])

    """ Add level of dropout defined in function call """
    if dropout:
        x = layers.SpatialDropout3D(dropout, name='pyramid_dropout')(x)

    """ Final convolution to produce channel for each filter """
    if n_classes == 1:
        activation = 'sigmoid'
    elif n_classes > 1:
        activation = 'softmax'
    outputs = Conv3DBn(n_classes, (1, 1, 1), activation=activation, kernel_initializer='he_normal', use_batchnorm=False, name='final')(x)

    model = models.Model(inputs=[inputs], outputs=[outputs])

    return model