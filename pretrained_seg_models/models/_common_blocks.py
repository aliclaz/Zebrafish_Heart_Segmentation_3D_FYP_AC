from keras_applications import get_submodules_from_kwargs

def Conv3DBn(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None,
             kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
             activity_regularizer=None, kernel_constraint=None, bias_constraint=None, use_batchnorm=False, **kwargs):
    conv_name, act_name, bn_name = None, None, None
    block_name = kwargs.pop('name', None)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if block_name is not None:
        conv_name = block_name + '_conv'

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(activation) else str(activation)
        act_name = block_name + '_' + act_str

    if block_name is not None and use_batchnorm:
        bn_name = block_name +'_bn'
    
    def wrapper(input_tensor):
        x = layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
                          dilation_rate=dilation_rate, activation=None, use_bias=not (use_batchnorm), 
                          kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, 
                          kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                          activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, 
                          bias_constraint=bias_constraint, name=conv_name)(input_tensor)
        
        if use_batchnorm:
            x = layers.BatchNormalization(axis=4, name=bn_name)(x)

        if activation:
            x = layers.Activation(activation, name=act_name)(x)

        return x
    
    return wrapper

def Conv3DTrans(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None,
                kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                activity_regularizer=None, kernel_constraint=None, bias_constraint=None, use_batchnorm=False, **kwargs):
    conv_trans_name, act_name, bn_name = None, None, None
    block_name = kwargs.pop('name', None)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if block_name is not None:
        transp_name = block_name + '_transp'

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(activation) else str(activation)
        act_name = block_name + '_' + act_str

    if block_name is not None and use_batchnorm:
        bn_name = block_name +'_bn'

    def wrapper(input_tensor):
        x = layers.Conv3DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
                          dilation_rate=dilation_rate, activation=None, kernel_initializer=kernel_initializer, 
                          bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                          activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, 
                          bias_constraint=bias_constraint, use_bias=not (use_batchnorm), name=transp_name)(input_tensor)
        
        if use_batchnorm:
            x = layers.BatchNormalization(axis=4, name=bn_name)(x)

        if activation:
            x = layers.Activation(activation, name=act_name)(x)

        return x
    
    return wrapper

def UpSamp3D(size=(2, 2, 2), data_format=None, **kwargs):
    up_samp_name = None
    block_name = kwargs.pop('name', None)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if block_name is not None:
        up_samp_name = block_name + '_up_samp'

    def wrapper(input_tensor):
        x = layers.UpSampling3D(size=size, data_format=data_format, name=up_samp_name)

        return x
    
    return wrapper

def AddAct(activation=None, **kwargs):
    add_name, act_name = None, None
    block_name = kwargs.pop('name', None)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if block_name is not None:
        add_name = block_name + '_add'

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(activation) else str(activation)
        act_name = block_name + '_' + act_str

    def wrapper(inputs):
        x = layers.add(inputs)
        x = layers.activation(activation)(x)

        return x
    
    return wrapper

def Mult(**kwargs):
    mult_name = None
    block_name = kwargs.pop('name', None)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if block_name is not None:
        mult_name = block_name + '_multiply'

    def wrapper(inputs):
        x = layers.multiply(inputs)

        return x
    
    return wrapper   