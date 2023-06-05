from keras_applications import get_submodules_from_kwargs

from ._common_blocks import Conv3DBn, Conv3DTrans
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

def DecoderBlock(filters, stage, use_batchnorm=False):
    kwargs = get_submodules()

    transp_block_name = 'decoder_stage{}a'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    def layer(input_tensor, skip=None):

        x = Conv3DTrans(filters, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding='same', use_batchnorm=use_batchnorm, 
                        name=transp_block_name, **kwargs)(input_tensor)

        if skip is not None:
            x = layers.Concatenate(axis=4, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer

def build_unet(backbone, skip_connection_layers, decoder_filters=(256, 128, 64, 32, 16), n_upsample_blocks=5, classes=1,
               activation='sigmoid', use_batchnorm=True, dropout=None):
    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling3D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)

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
    x = layers.Conv3D(filters=classes, kernel_size=(3, 3, 3), padding='same', use_bias=True, kernel_initializer='glorot_uniform',
                      name='final_conv',)(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model

def Unet(backbone_name='vgg16', input_shape=(None, None, 3), classes=1, activation='sigmoid', weights=None, encoder_weights='imagenet',
         encoder_freeze=False, encoder_features='default', decoder_filters=(256, 128, 64, 32, 16), decoder_use_batchnorm=True,
         dropout=None, **kwargs):
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    backbone = Backbones.get_backbone(backbone_name, input_shape=input_shape, weights=encoder_weights, include_top=False, **kwargs,)

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_unet(
        backbone=backbone,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
        dropout=dropout,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model