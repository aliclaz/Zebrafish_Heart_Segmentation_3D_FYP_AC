from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, Dropout, BatchNormalization, Activation

def encoder_block(input_layer, out_n_filters, drop_rate=None, max_pooling=True, use_batchnorm=False):
    conv = Conv3D(out_n_filters, kernel_size=(3, 3, 3), padding='same', kernel_initializer='HeNormal')(input_layer)
    if use_batchnorm:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    if drop_rate is not None:
        conv = Dropout(drop_rate)(conv)
    conv = Conv3D(out_n_filters, kernel_size=(3, 3, 3), padding='same', 
                  kernel_initializer='HeNormal')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    if max_pooling:
        next_layer = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv)
    else:
        next_layer = conv
    skip_connection = conv

    return next_layer, skip_connection

def decoder_block(conv_layer, concat_layer, out_n_filters, drop_rate=None, use_batchnorm=False):
    up = Conv3DTranspose(out_n_filters, (2, 2, 2), strides=(2, 2, 2), 
                         padding='same')(conv_layer)
    concat = concatenate([up, concat_layer], axis=3)
    conv = Conv3D(out_n_filters, (3, 3, 3), padding='same', activation='relu',
                  kernel_initializer='HeNormal')(concat)
    if use_batchnorm:
        conv = BatchNormalization()(conv)
    if drop_rate is not None:
        conv = Dropout(drop_rate)(conv)
    conv = Conv3D(out_n_filters, (3, 3, 3), padding='same', activation='relu',
                  kernel_initializer='HeNormal')(conv)

    return conv

def multi_unet_model(n_classes, input_shape=(None, None, None, 3), use_batchnorm=False, dropout=False):
    
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