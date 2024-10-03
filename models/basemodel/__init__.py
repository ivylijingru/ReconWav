from .basic_mlp import basicMLP


def get_base_model(cfg: dict):
    model = basicMLP(**cfg["args"])

    return model