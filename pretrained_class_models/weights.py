from . import get_submodules_from_kwargs

__all__ = ['load_model_weights']


def _find_weights(model_name, dataset, include_top):
    w = list(filter(lambda x: x['model'] == model_name, WEIGHTS_COLLECTION))
    w = list(filter(lambda x: x['dataset'] == dataset, w))
    w = list(filter(lambda x: x['include_top'] == include_top, w))

    return w

def load_model_weights(model, model_name, dataset, classes, include_top, **kwargs):
    _, _, _, keras_utils = get_submodules_from_kwargs(kwargs)

    weights = _find_weights(model_name, dataset, include_top)

    if weights:
        weights = weights[0]

        weights_path = keras_utils.get_file(weights['name'], weights['url'], cache_subdir='models', md5_hash=weights['md5'])

        model.load_weights(weights_path)

WEIGHTS_COLLECTION = [
    {
        'model': 'resnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/resnet18_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnet18_inp_channel_3_tch_0_top_False.h5',
        'md5': '1d04dd6c1f00b7bf4ba883c61bedeac8',
    },
    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/resnet34_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnet34_inp_channel_3_tch_0_top_False.h5',
        'md5': 'b7f3bcdc67c8614ba018c9fc5f75fc64',
    },
    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/resnet50_inp_channel_3_tch_0_top_False.h5',
        'name': 'resnet50_inp_channel_3_tch_0_top_False.h5',
        'md5': '2ba65fa189439a493ea8d1b22439ea2a',
    },
    {
        'model': 'vgg16',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0.4/vgg16_inp_channel_3_tch_0_top_False.h5',
        'name': 'vgg16_inp_channel_3_tch_0_top_False.h5',
        'md5': '240d399c45ed038a5a7b026d750ceb2b',
    }
]