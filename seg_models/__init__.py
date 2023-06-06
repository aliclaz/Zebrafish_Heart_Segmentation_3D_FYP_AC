import os
import functools
from . import base

_KERAS_FRAMEWORK_NAME = 'keras'
_TF_KERAS_FRAMEWORK_NAME = 'tf.keras'
_DEFAULT_KERAS_FRAMEWORK = _TF_KERAS_FRAMEWORK_NAME
_KERAS_FRAMEWORK = None
_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None
_KERAS_LOSSES = None


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils


def inject_global_losses(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['losses'] = _KERAS_LOSSES
        return func(*args, **kwargs)

    return wrapper


def inject_global_submodules(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = _KERAS_BACKEND
        kwargs['layers'] = _KERAS_LAYERS
        kwargs['models'] = _KERAS_MODELS
        kwargs['utils'] = _KERAS_UTILS
        return func(*args, **kwargs)

    return wrapper

def filter_kwargs(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_kwargs = {k: v for k, v in kwargs.items() if k in ['backend', 'layers', 'models', 'utils']}
        return func(*args, **new_kwargs)

    return wrapper

def framework():
    """Return name of Segmentation Models framework"""
    return _KERAS_FRAMEWORK

def set_framework(name):
    name = name.lower()

    if name == _KERAS_FRAMEWORK_NAME:
        import keras
    elif name == _TF_KERAS_FRAMEWORK_NAME:
        from tensorflow import keras
    else:
        raise ValueError('Not correct module name `{}`, use `{}` or `{}`'.format(
                name, _KERAS_FRAMEWORK_NAME, _TF_KERAS_FRAMEWORK_NAME))

    global _KERAS_BACKEND, _KERAS_LAYERS, _KERAS_MODELS
    global _KERAS_UTILS, _KERAS_LOSSES, _KERAS_FRAMEWORK

    _KERAS_FRAMEWORK = name
    _KERAS_BACKEND = keras.backend
    _KERAS_LAYERS = keras.layers
    _KERAS_MODELS = keras.models
    _KERAS_UTILS = keras.utils
    _KERAS_LOSSES = keras.losses

    # allow losses/metrics get keras submodules
    base.KerasObject.set_submodules(backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils,
    )

_framework = os.environ.get('SM_FRAMEWORK', _DEFAULT_KERAS_FRAMEWORK)
try:
    set_framework(_framework)
except ImportError:
    other = _TF_KERAS_FRAMEWORK_NAME if _framework == _KERAS_FRAMEWORK_NAME else _KERAS_FRAMEWORK_NAME
    set_framework(other)

print('Segmentation Models: using `{}` framework.'.format(_KERAS_FRAMEWORK))

# import helper modules
from . import losses
from . import metrics
from . import utils

# wrap segmentation models with framework modules
from .backbones.backbones_factory import Backbones
from .pretrained_seg_models.unet import Unet as _Unet
from .default_seg_models.def_unet import defUnet
from .pretrained_seg_models.atten_unet import AttentionUnet as _AttentionUnet
from .default_seg_models.def_atten_unet import defAttentionUnet
from .pretrained_seg_models.atten_res_unet import AttentionResUnet as _AttentionResUnet
from .default_seg_models.def_atten_res_unet import defAttentionResUnet

Unet = inject_global_submodules(_Unet)
AttentionUnet = inject_global_submodules(_AttentionUnet)
AttentionResUnet = inject_global_submodules(_AttentionResUnet)
get_available_backbone_names = Backbones.models_names

def get_preprocessing(name):
    preprocess_input = Backbones.get_preprocessing(name)
    preprocess_input = inject_global_submodules(preprocess_input)
    preprocess_input = filter_kwargs(preprocess_input)

    return preprocess_input
    
__all__ = ['get_submodules_from_kwargs', 'Unet', 'defUnet', 'AttentionUnet', 'defAttentionUnet', 'AttentionResUnet', 'set_framework',
           'framework', 'get_preprocessing', 'get_available_backbone_names', 'utils']