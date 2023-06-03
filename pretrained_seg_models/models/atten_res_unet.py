from keras_applications import get_submodules_from_kwargs

from ._common_blocks import Conv3DBn, Conv3DTrans, UpSamp3D, AddAct, Mult
from ._utils import freeze_model
from ..backbones.backbones_factory import Backbones

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
        x = Conv3x3BnReLU(filters, use_batchnorm, name=name)(input_tensor)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=name)(x)
        shortcut = Conv3DBn(filters, 1, kernel_initializer='he_uniform', padding='same', use_batchnorm=use_batchnorm, name=name, **kwargs)(input_tensor)
        x = AddAct('relu', name=name, **kwargs)([shortcut, x])

        return x
    return wrapper

def RepeatElement(tensor, rep, name=None):
    return layers.Lambda(lambda x, repnum: backend.repeat_elements(x, repnum, axis=4), arguments={'repnum': rep},
                         name=name)(tensor)     

def GatingSignal(filters, use_batchnorm, name=None):
    kwargs = get_submodules()
    
    def wrapper(input_tensor):
        return Conv3DBn(filters, kernel_size=(1, 1, 1), activation='relu', padding='same', kernel_initializer='he_uniform', 
                        use_batchnorm=use_batchnorm, name=name, **kwargs)(input_tensor)
    
    return wrapper

def AttentionBlock(inter_shape, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(skip_connection, gating):
        shape_x = backend.int_shape(skip_connection)
        shape_g = backend.int_shape(gating)

        theta_x = Conv3DBn(inter_shape, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal',
                           use_batchnorm=use_batchnorm, name=name, **kwargs)(skip_connection)
        shape_theta_x = backend.int_shape(theta_x)

        phi_g = Conv3DBn(inter_shape, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal',
                         use_batchnorm=use_batchnorm, name=name, **kwargs)(gating)
        upsample_g = Conv3DTrans(inter_shape, (3, 3, 3), padding='same', strides=(shape_theta_x[1] // shape_g[1],
                                                                                 shape_theta_x[2] // shape_g[2],
                                                                                 shape_theta_x[3] // shape_g[3]),
                                                                                 name=name, **kwargs)(phi_g)
        
        act_xg = AddAct('relu', name=name, **kwargs)([upsample_g, theta_x])
        sigmoid_xg = Conv3DBn(1, kernel_size=(1, 1, 1), activation='softmax', kernel_initializer='he_normal', padding='same',
                              use_batchnorm=use_batchnorm, name=name, **kwargs)(act_xg)
        shape_sigmoid = backend.int_shape(sigmoid_xg)
        upsample_psi = UpSamp3D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3]), 
                                name=name, **kwargs)(sigmoid_xg)
        upsample_psi = RepeatElement(upsample_psi, shape_x[4], name=name)

        y = Mult()([upsample_psi, skip_connection])

        result = Conv3DBn(shape_x[4], (1, 1, 1), kernel_intializer='he_normal', padding='same', use_batchnorm=True, name=name, **kwargs)(y)
        
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
        x = UpSamp3D(size=(), name=up_name, **kwargs)(input_tensor)
        if skip is not None:
            x = layers.Concatenate(axis=4, name=concat_name)([x, atten])
        
        x = ResConvBlock(filters, use_batchnorm, name=res_conv_block_name)(x)

        return x

    return layer

def build_atten_res_unet(backbone, skip_connection_layers, decoder_filters=(256, 128, 64, 32, 16), n_upsample_blocks=5, classes=1, activation='sigmoid', 
                         use_batchnorm=True, dropout=None,):
    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling3D):
        x = ResConvBlock(512, use_batchnorm, name='centre_block')(x)
        # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = DecoderBlock(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)

    if dropout:
        x = layers.SpatialDropout3D(dropout, name='pyramid_dropout')(x)

    # model head (define number of output classes)
    x = layers.Conv3D(filters=classes, kernel_size=(3, 3, 3), padding='same', use_bias=True, kernel_initializer='glorot_uniform', name='final_conv',)(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model

def AttentionResUnet(backbone_name='vgg16', input_shape=(None, None, None, 3), classes=1, activation='sigmoid', weights=None, encoder_weights='imagenet', 
                     encoder_freeze=False, encoder_features='default', decoder_filters=(256, 128, 64, 32, 16,), decoder_use_batchnorm=True, dropout=None, **kwargs):
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    backbone = Backbones.get_backbone(backbone_name, input_shape=input_shape, weights=encoder_weights, include_top=False, **kwargs)

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_atten_res_unet(backbone=backbone, skip_connection_layers=encoder_features, decoder_filters=decoder_filters, n_upsample_blocks=len(decoder_filters), 
                                 classes=classes, activation=activation, use_batchnorm=decoder_use_batchnorm, dropout=dropout,)
    
    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model