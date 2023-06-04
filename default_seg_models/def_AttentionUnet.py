from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, Dropout, BatchNormalization, Activation, \
add, UpSampling3D, multiply, Lambda
import keras.backend as K

def encoder_block(out_n_filters, drop_rate=None, max_pooling=True, use_batchnorm=False):
    def wrapper(input_layer):
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
    
    return wrapper

def RepeatElement(tensor, rep):
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4), arguments={'repnum': rep})(tensor)   

def GatingSignal(filters, use_batchnorm):
    def wrapper(input_tensor):
        x = Conv3D(filters, kernel_size=(1, 1, 1), activation='relu', padding='same', kernel_initializer='he_uniform')(input_tensor)
        if use_batchnorm:
            x = BatchNormalization()(x)
        
        return x
    
    return wrapper

def AttentionBlock(inter_shape):
    def wrapper(skip_connection, gating):
        shape_x = K.int_shape(skip_connection)
        shape_g = K.int_shape(gating)

        theta_x = Conv3D(inter_shape, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initalizer='he_normal')(skip_connection)
        shape_theta_x = K.int_shape(theta_x)
        phi_g = Conv3D(inter_shape, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal')(gating)
        upsample_g = Conv3DTranspose(inter_shape, (3, 3, 3), padding='same', strides=(shape_theta_x[1] // shape_g[1],
                                                                                      shape_theta_x[2] // shape_g[2],
                                                                                      shape_theta_x[3] // shape_g[3]))(phi_g)
        
        concat_xg = add([upsample_g, theta_x])
        act_xg = Activation('relu')(concat_xg)
        psi = Conv3D(1, kernel_size=(1, 1, 1), activation='softmax', kernel_initializer='he_normal', padding='same')(act_xg)
        sigmoid_xg = Activation('softmax')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = UpSampling3D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2], 
                                          shape_x[3] // shape_sigmoid[3]))(sigmoid_xg)
        upsample_psi = RepeatElement(upsample_psi, shape_x[4])

        y = multiply([upsample_psi, skip_connection])

        result = Conv3D(shape_x[4], (1, 1, 1), kernel_intitializer='he_normal', padding='same')(y)
        
        return result
    
    return wrapper

def decoder_block(out_n_filters, drop_rate=None, use_batchnorm=False):
    def wrapper(conv_layer, concat_layer):
        x = GatingSignal(out_n_filters, use_batchnorm)(conv_layer)
        atten = AttentionBlock(out_n_filters, use_batchnorm)(concat_layer, x)
        x = concatenate([x, atten], axis=4)
        x = Conv3D(out_n_filters, (3, 3, 3), kernel_intializer='he_normal', padding='same')
        if use_batchnorm:
            

        return x

    return wrapper

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