from .basic_conv_multi_encodec import basicConvENCODEC
from .basic_conv_multi_mert import basicConvMERT


def get_base_model(cfg: dict):
    if cfg["args"]["embed_type"] == "mert":
        model = basicConvMERT(**cfg["args"])
    elif cfg["args"]["embed_type"] == "encodec":
        model = basicConvENCODEC(**cfg["args"])

    return model
