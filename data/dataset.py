import json
import os
import numpy as np

import torch
import torch.utils.data as Data


class ReconstrcutDataset(Data.Dataset):
    def __init__(
        self,
        manifest_path,
        mel_frame_rate,
        input_sec,
    ) -> None:
        super().__init__()

        with open(manifest_path) as f:
            self.data = [json.loads(line) for line in f]

        self.target_seq_len = mel_frame_rate * input_sec
        self.input_sec = input_sec

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        mel
        vgg
        """
        output_data = dict()
        mel = torch.from_numpy(np.load(os.path.join("preprocess", self.data[idx]["mel_path"])))
        output_data["mel"] = torch.zeros(mel.shape[0], self.target_seq_len)
        if mel.shape[1] < self.target_seq_len:
            output_data["mel"][:, :mel.shape[1]] = mel
        elif mel.shape[1] >= self.target_seq_len:
            output_data["mel"] = mel[:, :self.target_seq_len]

        output_data["vggish"] = torch.from_numpy(np.load(os.path.join("preprocess", self.data[idx]["vggish_path"])))
        return output_data

