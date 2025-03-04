import torch.utils.data as Data
import pytorch_lightning as pl

from .dataset import (
    ReconstrcutDatasetMERT,
    ReconstrcutDatasetENCODEC,
    ReconstructionDatasetMERTCodebook,
)


class ReconstructDataModule(pl.LightningDataModule):
    def __init__(
        self,
        embed_type,
        train_manifest_path,
        val_manifest_path,
        test_manifest_path,
        mert_version,
        mel_frame_rate,
        input_sec,
        batch_size,
        train_shuffle,
        num_workers,
    ) -> None:
        super().__init__()

        # switch function according to the embed_type

        # build a dictionary to map the embed_type to the corresponding dataset
        dataset_dict = {
            "mert": ReconstrcutDatasetMERT,
            "encodec": ReconstrcutDatasetENCODEC,
            "mert_cb0": ReconstructionDatasetMERTCodebook,
        }

        if embed_type not in dataset_dict:
            raise ValueError(f"embed_type {embed_type} not supported")

        self.train_dataset = dataset_dict[embed_type](
            train_manifest_path, mel_frame_rate, input_sec, mert_version
        )
        self.val_dataset = dataset_dict[embed_type](
            val_manifest_path, mel_frame_rate, input_sec, mert_version
        )
        self.test_dataset = dataset_dict[embed_type](
            test_manifest_path, mel_frame_rate, input_sec, mert_version
        )

        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers

    def train_dataloader(self):
        return Data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return Data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return Data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
