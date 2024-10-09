"""
Convert a tensor with shape [batch_size, seq_len_latent, shape_latent]
We want to convert to shape [batch_size, seq_len_feature, shape_feature]

What we can do is:
    First interpolate seq_len_latent to seq_len_feature (e.g. adaptive average pooling)
    Then do self-attention on latent embedding
    (This is similar to what FastSpeech is doing after )

Alternative:
    We use cross-attention and predict the features recursively
    First do self-attention on latent embedding
    (This idea tends to formulate the problem similar to translation)

TODO: WIP; will first implement a shorter version of things
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class basicConv(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        target_seq_len, # this guy is around 344 for 4 seconds
    ) -> None:
        super().__init__()
        self.target_seq_len = target_seq_len
        # 计算卷积核大小和步幅，确保输入的 time_stamp1 能够映射到输出的 time_stamp2
        padding = 0      # 可以根据需求调整
        output_padding = 0 # 当需要精准控制输出尺寸时，可以设置

        stride1 = 2      # 可以根据需求调整
        self.deconv1 = nn.ConvTranspose1d(
            in_channels=input_dim, 
            out_channels=output_dim, 
            kernel_size=2 * stride1, 
            stride=stride1, 
            padding=padding, 
            output_padding=output_padding
        )
        self.batch_norm1 = nn.BatchNorm1d(output_dim)
        self.activation1 = nn.ReLU()

        stride2 = 4
        self.deconv2 = nn.ConvTranspose1d(
            in_channels=output_dim, 
            out_channels=output_dim, 
            kernel_size=2 * stride2, 
            stride=stride2, 
            padding=1, 
            output_padding=output_padding
        )
        self.batch_norm2 = nn.BatchNorm1d(output_dim)
        self.activation2 = nn.ReLU()

        stride3 = 8
        self.deconv3 = nn.ConvTranspose1d(
            in_channels=output_dim, 
            out_channels=output_dim, 
            kernel_size=2 * stride3, 
            stride=stride3, 
            padding=0, 
            output_padding=output_padding
        )
        self.batch_norm3 = nn.BatchNorm1d(output_dim)
        self.activation3 = nn.ReLU()

    def forward(self, x):
        x = x.transpose(-2, -1) # [batch_size, shape_latent, seq_len_latent]

        x = self.activation1(self.batch_norm1(self.deconv1(x)))
        x = self.activation2(self.batch_norm2(self.deconv2(x)))
        output = self.activation3(self.batch_norm3(self.deconv3(x)))

        return output # [batch_size, shape_latent, seq_len_target]


if __name__ == "__main__":
    input_dim = 128
    output_dim = 80
    target_seq_len = 344

    model = basicConv(input_dim, output_dim, target_seq_len)
    x = torch.zeros(16, 4, 128)
    print(model(x).shape)