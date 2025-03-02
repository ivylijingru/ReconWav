from .basic_conv_multi_encodec import basicConvENCODEC
from .basic_conv_multi_mert import basicConvMERT
from .basic_conv_multi_mert_cb0 import basicConvMERT as basicConvMERTcb0


def get_base_model(cfg: dict):
    if cfg["embed_type"] == "mert":
        model = basicConvMERT(**cfg["args"])
    elif cfg["embed_type"] == "encodec":
        model = basicConvENCODEC(**cfg["args"])
    elif cfg["embed_type"] == "mert_cb0":
        model = basicConvMERTcb0(**cfg["args"])

    return model
