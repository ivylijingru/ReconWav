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

import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import SConv1d, NormConvTranspose1d


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.
    Args:
        dim (int): Dimension of the input/output
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3)
        true_skip (bool): Whether to use true skip connection or a simple convolution as the skip connection.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: tp.List[int] = [3, 1],
        dilations: tp.List[int] = [1, 1],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "weight_norm",
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(
            dilations
        ), "Number of kernel sizes should match number of dilations"
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                SConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class basicConvENCODEC(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        target_seq_len,  # this guy is around 344 for 4 seconds
    ) -> None:
        super().__init__()
        self.target_seq_len = target_seq_len
        # 计算卷积核大小和步幅，确保输入的 time_stamp1 能够映射到输出的 time_stamp2
        padding = 0  # 可以根据需求调整
        output_padding = 0  # 当需要精准控制输出尺寸时，可以设置

        stride1 = 1  # 可以根据需求调整

        activation: str = "ELU"
        activation_params: dict = {"alpha": 1.0}
        norm: str = "weight_norm"
        norm_params: tp.Dict[str, tp.Any] = {}
        residual_kernel_size: int = 3
        dilation_base: int = 2
        causal: bool = False
        pad_mode: str = "constant"
        true_skip: bool = False
        compress: int = 2

        self.deconv1 = NormConvTranspose1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=23 * stride1,
            stride=stride1,
            padding=padding,
            output_padding=output_padding,
            norm=norm,
        )

        self.res_block1 = SEANetResnetBlock(
            output_dim,
            kernel_sizes=[residual_kernel_size, 1],
            dilations=[dilation_base**1, 1],
            activation=activation,
            activation_params=activation_params,
            norm=norm,
            norm_params=norm_params,
            causal=causal,
            pad_mode=pad_mode,
            compress=compress,
            true_skip=true_skip,
        )

        self.deconv2 = NormConvTranspose1d(
            in_channels=output_dim,
            out_channels=output_dim,
            kernel_size=23 * stride1,
            stride=stride1,
            padding=padding,
            output_padding=output_padding,
            norm=norm,
        )

        self.res_block2 = SEANetResnetBlock(
            output_dim,
            kernel_sizes=[residual_kernel_size, 1],
            dilations=[dilation_base**1, 1],
            activation=activation,
            activation_params=activation_params,
            norm=norm,
            norm_params=norm_params,
            causal=causal,
            pad_mode=pad_mode,
            compress=compress,
            true_skip=true_skip,
        )
        self.activation = nn.ELU()

    def forward(self, x):
        # [batch_size, shape_latent, seq_len_latent]
        output = self.deconv1(x)
        output = self.res_block1(output)
        output = self.activation(output)
        output = self.deconv2(output)
        output = self.res_block2(output)
        return output  # [batch_size, shape_latent, seq_len_target]


if __name__ == "__main__":
    input_dim = 128
    output_dim = 80
    target_seq_len = 344

    model = basicConvENCODEC(input_dim, output_dim, target_seq_len)
    x = torch.zeros(16, 128, 300)
    print(model(x).shape)
