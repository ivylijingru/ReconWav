import json

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data import ReconstructDataModule
from models import ModelFactory

import torch


def forward_hook(module, input, output):
    # 检查前向传播中的 NaN（兼容元组输入）

    def check_nan(tensor, name):
        if tensor is not None and torch.is_tensor(tensor):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print("Forward")
                print("nan ", torch.isnan(tensor).any())
                print("inf ", torch.isinf(tensor).any())
                print(f"NaN/Inf 出现在模块: {module.__class__.__name__} 的 {name}")
                print("具体张量形状:", tensor.shape)

                print(input)
                print(module.weight)
                print(output)
                raise ValueError("检测到 NaN/Inf，终止训练")

    if hasattr(module, "weight"):
        if module.weight is not None:

            weight = module.weight
            if torch.isnan(weight).any():
                print("Forward")
                print(f"NaN 出现在 {module} 的权重中！")
                print("权重统计:", weight.min(), weight.mean(), weight.max())

                import numpy as np

                # np.save("input.npy", input.detach().cpu().numpy())
                np.save("module weight.npy", module.weight.detach().cpu().numpy())
                np.save("output.npy", output.detach().cpu().numpy())

                raise ValueError("权重包含 NaN")

    # 检查输入（input 是一个包含多个张量的元组）
    if isinstance(input, tuple):
        for i, inp in enumerate(input):
            check_nan(inp, f"input[{i}]")
    else:
        check_nan(input, "input")

    # 检查输出（output 可能是张量或元组）
    if isinstance(output, tuple):
        for i, out in enumerate(output):
            check_nan(out, f"output[{i}]")
    else:
        check_nan(output, "output")


def backward_hook(module, grad_input, grad_output):
    # 检查反向传播中的 NaN 梯度（兼容元组输入）
    def check_nan(tensor, name):
        if tensor is not None and torch.is_tensor(tensor):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print("Backward")
                print(f"NaN/Inf 梯度出现在模块: {module.__class__.__name__} 的 {name}")
                print("具体梯度形状:", tensor.shape)
                raise ValueError("检测到 NaN/Inf 梯度，终止训练")

    if hasattr(module, "weight"):
        if module.weight is not None:

            weight = module.weight
            if torch.isnan(weight).any():
                print("Backward")
                print(f"NaN 出现在 {module} 的权重中！")
                print("权重统计:", weight.min(), weight.mean(), weight.max())

                print(input)
                print(module.weight)
                print(output)
                raise ValueError("权重包含 NaN")

    # 检查梯度输入（grad_input 是元组）
    if isinstance(grad_input, tuple):
        for i, gin in enumerate(grad_input):
            check_nan(gin, f"grad_input[{i}]")
    else:
        check_nan(grad_input, "grad_input")

    # 检查梯度输出（grad_output 是元组）
    if isinstance(grad_output, tuple):
        for i, gout in enumerate(grad_output):
            check_nan(gout, f"grad_output[{i}]")
    else:
        check_nan(grad_output, "grad_output")


def train(config):
    with open(config) as f:
        config = json.load(f)

    pl.seed_everything(config["seed"], workers=True)

    data_cfg = config["data"]
    model_cfg = config["model"]
    trainer_cfg = config["trainer"]

    datamodule = ReconstructDataModule(**data_cfg)
    model = ModelFactory.create_model(data_cfg["embed_type"], model_cfg)

    # 注册前向钩子
    for name, layer in model.named_modules():
        layer.register_forward_hook(forward_hook)

    # 注册反向钩子（可选）
    for name, layer in model.named_modules():
        layer.register_backward_hook(backward_hook)

    callbacks = [
        ModelCheckpoint(**trainer_cfg["checkpoint"]),
        # EarlyStopping(**trainer_cfg["early_stopping"]),
    ]

    trainer = pl.Trainer(
        **trainer_cfg["args"],
        logger=TensorBoardLogger(**trainer_cfg["logger"]),
        callbacks=callbacks,
        resume_from_checkpoint="work_dir_mert/weight_new_data_cb0_lr0/epoch=0-val_loss-total=2.623.ckpt",
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule)


if __name__ == "__main__":
    import fire

    fire.Fire(train)
