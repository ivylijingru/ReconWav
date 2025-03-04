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


class basicConvMERT(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        target_seq_len,  # Target sequence length (300)
    ) -> None:
        super().__init__()
        self.target_seq_len = target_seq_len

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

        # 主要修改部分
        # Stage 1: 将长度从199扩展到300
        self.deconv1 = NormConvTranspose1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=34,  # 精确控制输出长度
            stride=1,
            padding=0,
            output_padding=0,
            norm=norm,
        )

        self.res_block1 = SEANetResnetBlock(
            input_dim,
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

        # Stage 2: 保持长度300
        self.deconv2 = NormConvTranspose1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=35,  # 保持长度的卷积参数
            stride=1,
            padding=0,  # 保持长度关键参数
            output_padding=0,
            norm=norm,
        )

        self.res_block2 = SEANetResnetBlock(
            input_dim,
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

        # Stage 3: 通道数调整到1024
        self.deconv3 = NormConvTranspose1d(
            in_channels=input_dim,
            out_channels=output_dim,  # 最终输出通道1024
            kernel_size=35,
            stride=1,
            padding=0,
            output_padding=0,
            norm=norm,
        )

        self.res_block3 = SEANetResnetBlock(
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

        self.activation1 = nn.ELU()
        self.activation2 = nn.ELU()

    def forward(self, x):
        # [batch_size, shape_latent, seq_len_latent]
        output = self.deconv1(x)  # [16, 768, 300]
        output = self.res_block1(output)
        output = self.activation1(output)

        output = self.deconv2(output)  # [16, 768, 300]
        output = self.res_block2(output)
        output = self.activation2(output)

        output = self.deconv3(output)  # [16, 1024, 300]
        output = self.res_block3(output)
        return output


class avgPoolConvMert(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        target_seq_len,  # Target sequence length (300)
    ) -> None:
        super().__init__()
        self.target_seq_len = target_seq_len
        self.avg_pool = nn.AdaptiveAvgPool1d(self.target_seq_len)
        
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

        input_dim = self.target_seq_len
        output_dim = self.target_seq_len

        # 主要修改部分
        # Stage 1: 将长度从199扩展到300
        self.deconv1 = NormConvTranspose1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=86,  # 精确控制输出长度
            stride=1,
            padding=0,
            output_padding=0,
            norm=norm,
        )

        self.res_block1 = SEANetResnetBlock(
            input_dim,
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

        # Stage 2: 保持长度300
        self.deconv2 = NormConvTranspose1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=86,  # 保持长度的卷积参数
            stride=1,
            padding=0,  # 保持长度关键参数
            output_padding=0,
            norm=norm,
        )

        self.res_block2 = SEANetResnetBlock(
            input_dim,
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

        # Stage 3: 通道数调整到1024
        self.deconv3 = NormConvTranspose1d(
            in_channels=input_dim,
            out_channels=output_dim,  # 最终输出通道1024
            kernel_size=87,
            stride=1,
            padding=0,
            output_padding=0,
            norm=norm,
        )

        self.res_block3 = SEANetResnetBlock(
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

        self.activation1 = nn.ELU()
        self.activation2 = nn.ELU()

    def forward(self, x):
        output = self.avg_pool(x) # torch.Size([16, 768, 300])
        # print(output.shape)
        output = output.permute(0, 2, 1) # torch.Size([16, 300, 768])
        # print(output.shape)
        output = self.deconv1(output)  # [16, 768, 300]
        output = self.res_block1(output)
        output = self.activation1(output)

        output = self.deconv2(output)  # [16, 768, 300]
        output = self.res_block2(output)
        output = self.activation2(output)

        output = self.deconv3(output)  # [16, 1024, 300]
        output = self.res_block3(output)
        output = output.permute(0, 2, 1)
        return output 

if __name__ == "__main__":
    input_dim = 768
    output_dim = 1024
    target_seq_len = 300

    model = basicConvMERT(input_dim, output_dim, target_seq_len)
    x = torch.zeros(16, 768, 199)
    # 1024 * 300
    print(model(x).shape)

    model = avgPoolConvMert(input_dim, output_dim, target_seq_len)
    x = torch.zeros(16, 768, 199)
    # 1024 * 300
    print(model(x).shape)