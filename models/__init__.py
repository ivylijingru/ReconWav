# models/__init__.py
from importlib import import_module

from model_factory.encodec_model import ReconstructModelENCODEC
from model_factory.mert_model import ReconstructModelMERT


class ModelFactory:
    @staticmethod
    def create_model(embed_type: str, config: dict):
        """
        根据类型加载不同模型的实现
        Usage: model = ModelFactory.create_model("mert", in_dim=256)
        """
        try:
            if embed_type == "mert":
                return ReconstructModelMERT(config)
            elif embed_type == "encodec":
                return ReconstructModelENCODEC(config)
        except ImportError:
            raise ValueError(f"Unsupported model type: {embed_type}")
