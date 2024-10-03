import torch.utils.data as Data
import pytorch_lightning as pl

from .dataset import ReconstrcutDataset


class ReconstructDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_manifest_path,
        val_manifest_path,
        test_manifest_path,
        mel_frame_rate,
        input_sec,
        batch_size,
        train_shuffle,
        num_workers,
    ) -> None:
        super().__init__()

        self.train_dataset = ReconstrcutDataset(train_manifest_path, mel_frame_rate, input_sec)
        self.val_dataset = ReconstrcutDataset(val_manifest_path, mel_frame_rate, input_sec)
        self.test_dataset = ReconstrcutDataset(test_manifest_path, mel_frame_rate, input_sec)

        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers

    def train_dataloader(self):
        return Data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return Data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return Data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)