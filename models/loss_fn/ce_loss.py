import torch
import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()  # 改为交叉熵损失

    def forward(self, logits, target_indices):
        loss_dict = dict()
        # 输入形状检查:
        # logits应为 [batch, num_cls=1024, seq_len]
        # target_indices应为 [batch, seq_len] 且 dtype=long
        logits = logits.permute(0, 2, 1)
        logits = torch.flatten(logits, end_dim=-2)
        target_indices = torch.flatten(target_indices)
        ce_loss = self.ce_loss(logits, target_indices)
        loss_dict["loss/ce"] = ce_loss
        return loss_dict
