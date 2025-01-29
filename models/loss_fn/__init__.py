from .recon_loss import ReconLoss


def get_loss_fn():
    return ReconLoss()
