from .basic_mlp import basicMLP
from .basic_conv import basicConv


def get_base_model(cfg: dict):
    model = basicConv(**cfg["args"])

    return model