"""
Implement a mel-spectrogram MSE loss here
"""

import torch
import torch.nn as nn


class ReconLoss(nn.Module):
    def __init__(self):
        super(ReconLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, mel, mel_target):
        loss_dict = dict()

        mel_target.requires_grad = False
        mel_loss = self.mse_loss(mel, mel_target)

        loss_dict["loss/mel"] = mel_loss
        return loss_dict