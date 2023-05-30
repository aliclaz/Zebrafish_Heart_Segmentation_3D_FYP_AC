import tensorflow.keras as tfkeras
from .models_factory import ModelFactory

@staticmethod
class TFKerasModelsFactory(ModelsFactory):
    def get_kwargs():
        return {'backend': tfkeras.backend, 'layers': tfkeras.layers, 'models': tfkeras.models, 'utils': tfkeras.utils}
    
Classifiers = TFKerasModelsFactory()