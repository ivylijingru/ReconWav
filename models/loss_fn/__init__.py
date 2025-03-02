from .recon_loss import ReconLoss
from .ce_loss import CELoss


def get_loss_fn(cfg: dict):
    if cfg["type"] == "ce":
        return CELoss()
    elif cfg["type"] == "mse":
        return ReconLoss()
