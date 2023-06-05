from . import _KERAS_BACKEND, _KERAS_LAYERS, _KERAS_UTILS, _KERAS_MODELS

def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', ka._KERAS_BACKEND)
    layers = kwargs.get('layers', ka._KERAS_LAYERS)
    models = kwargs.get('models', ka._KERAS_MODELS)
    utils = kwargs.get('utils', ka._KERAS_UTILS)

    return backend, layers, models, utils