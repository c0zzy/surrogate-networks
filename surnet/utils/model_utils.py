from models.deep_gp import DeepGPSurrogate
from models.gp import GPSurrogate


def build_model(hps, **kwargs):
    name = hps['model']['name']
    model_class = {
        'gp': GPSurrogate,
        'deep_gp': DeepGPSurrogate,
    }[name]

    return model_class(hps, **kwargs)