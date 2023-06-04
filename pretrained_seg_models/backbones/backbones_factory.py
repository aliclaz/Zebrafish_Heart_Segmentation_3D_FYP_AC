import copy
from ..pretrained_class_models.models_factory import ModelsFactory

class BackbonesFactory(ModelsFactory):
    _default_feature_layers = {
        'vgg16': ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
        'resnet18': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'resnet34': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'resnet50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0')
    }

    _models_update = {}

    _models_delete = []

    @property
    def models(self):
        all_models = copy.copy(self._models)
        all_models.update(self._models_update)
        for k in self._models_delete:
            del all_models[k]
        return all_models
    
    def get_backbone(self, name, *args, **kwargs):
        model_fn, _ = self.get(name)
        model = model_fn(*args, **kwargs)

        return model
    
    def get_feature_layers(self, name, n=5):
        return self._default_feature_layers[name][:n]
    
    def get_preprocessing(self, name):
        return self.get(name)[1]
    
Backbones = BackbonesFactory()