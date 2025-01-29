from .basic_conv_multi_encodec import basicConvMERT
from .basic_conv_multi_mert import basicConvENCODEC


def get_base_model(cfg: dict):
    if cfg["name"] == "mert":
        model = basicConvMERT(**cfg["args"])
    elif cfg["name"] == "encodec":
        model = basicConvENCODEC(**cfg["args"])

    return model
