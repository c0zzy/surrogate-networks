import yaml
import pathlib


def load_config(env='local'):
    filename = f'config_{env}.yaml'
    path = pathlib.Path(__file__).parent / filename
    with open(path, "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
        return cfg
