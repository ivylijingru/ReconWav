from typing import Any

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoModel
from transformers import Wav2Vec2FeatureExtractor

from ..basemodel import get_base_model
from ..loss_fn import get_loss_fn


class ReconstructModelMERT(pl.LightningModule):
    def __init__(self, configs) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.optim_cfg = configs["optim"]
        self.model = get_base_model(configs["conv"])
        self.loss_fn = get_loss_fn(configs["loss"])
        self.example_batch = None

        self.mert_model = AutoModel.from_pretrained(
            "m-a-p/MERT-v0-public", trust_remote_code=True
        )
        self.mert_model.eval()

    def on_train_epoch_start(self):
        self.mert_model.config.mask_time_prob = 0.0
        for param in self.mert_model.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx) -> Any:
        loss_dict, _ = self.common_step(batch)

        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"])
        self.log_dict_prefix(loss_dict, "train")

        # self.train_metrics.update(logits, torch.round(batch["y"]), batch["y_mask"])

        return loss_dict["loss/total"]

    def validation_step(self, batch, batch_idx) -> Any:
        loss_dict, mel_output = self.common_step(batch)

        if batch_idx == 0:
            self.example_batch = batch
            self.example_batch["mel_pred"] = mel_output
        self.log_dict_prefix(loss_dict, "val")

        # self.val_metrics.update(logits, torch.round(batch["y"]), batch["y_mask"])

        return loss_dict["loss/total"]

    # def on_validation_epoch_end(self):
    #     # TODO: there is bug here; needs debugging
    #     if self.example_batch is not None:
    #         for i in range(self.example_batch["mel"].shape[0] // 8):
    #             self.log_image(self.example_batch["mel"][i], "gt_{}".format(i))
    #             self.log_image(self.example_batch["mel_pred"][i], "pred_{}".format(i))

    def test_step(self, batch, batch_idx):
        loss_dict, _ = self.common_step(batch)

        self.log_dict_prefix(loss_dict, "test")

        # self.test_metrics.update(logits, torch.round(batch["y"]), batch["y_mask"])

    def common_step(self, batch):
        inputs = batch["inputs"]
        mel = batch["mel"]

        loss_dict = dict()

        with torch.no_grad():
            outputs = self.mert_model(**inputs, output_hidden_states=True)
        model_output = outputs.hidden_states[-1]
        model_output = model_output.transpose(-1, -2)
        # print(model_output.shape)
        # print(encodec.shape)
        model_output = self.model(model_output)
        loss_dict = self.loss_fn(model_output, mel)

        total_loss = 0
        for loss_key in loss_dict.keys():
            total_loss += loss_dict[loss_key]
        loss_dict["loss/total"] = total_loss

        # TODO: potentially log mel spectrogram here
        return loss_dict, model_output

    def inference_step(self, audio):
        # process and extract embeddings
        # Assume that audio is already resampled
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v0-public", trust_remote_code=True
        )
        resample_rate = processor.sampling_rate
        inputs = processor(audio, sampling_rate=resample_rate, return_tensors="pt")

        for input_key in inputs.keys():
            inputs[input_key] = inputs[input_key].squeeze(0).cuda()

        with torch.no_grad():
            outputs = self.mert_model(**inputs, output_hidden_states=True)
        model_output = outputs.hidden_states[-1]
        model_output = model_output.transpose(-1, -2)

        model_output = self.model(model_output)

        return model_output

    def log_dict_prefix(self, d, prefix):
        for k, v in d.items():
            self.log("{}/{}".format(prefix, k), v)

    def log_image(self, mel_spectrogram, note):
        # 将频谱图转换为 NumPy 数组，以便绘制
        mel_spectrogram_np = mel_spectrogram.detach().cpu().numpy()

        # 使用 Matplotlib 将梅尔频谱图转换为彩色图像
        fig, ax = plt.subplots()
        cax = ax.imshow(
            mel_spectrogram_np, aspect="auto", cmap="viridis"
        )  # 选择一个彩色 colormap，比如 viridis
        fig.colorbar(cax)

        # 将 Matplotlib 图像保存到内存中
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # 记录到 TensorBoard
        self.logger.experiment.add_image(
            "MelSpectrogram_{}".format(note), img, self.current_epoch, dataformats="HWC"
        )

        # 关闭 Matplotlib 图像以释放内存
        plt.close(fig)

    def configure_optimizers(self) -> Any:
        optimizer_cfg = self.optim_cfg["optimizer"]
        scheduler_cfg = self.optim_cfg["scheduler"]

        optimizer = torch.optim.__dict__.get(optimizer_cfg["name"])(
            self.parameters(), **optimizer_cfg["args"]
        )
        scheduler = torch.optim.lr_scheduler.__dict__.get(scheduler_cfg["name"])(
            optimizer, **scheduler_cfg["args"]
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                monitor=scheduler_cfg["monitor"],
            ),
        )

    def on_after_backward(self):
        # 在 backward 之后调用，用于监控梯度
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.mean()
                grad_max = param.grad.max()
                grad_min = param.grad.min()
                self.log(f"grad_mean/{name}", grad_mean)
                self.log(f"grad_max/{name}", grad_max)
                self.log(f"grad_min/{name}", grad_min)
