import json
import os

import torch.utils.data as Data

from data.data_utils import process_mert_format, preprocess_encodec_format
from data.data_utils import preprocess_mel


class ReconstrcutDatasetMERT(Data.Dataset):
    def __init__(
        self,
        manifest_path,
        mel_frame_rate,
        input_sec,
    ) -> None:
        super().__init__()

        with open(manifest_path) as f:
            self.data = [json.loads(line) for line in f]

        self.target_seq_len = int(mel_frame_rate * input_sec)
        self.manifest_path = manifest_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return:
            output_data: dict
                {
                    "inputs": dict,
                    "mel": torch.tensor
                }
        """
        output_data = dict()

        audio_path = os.path.join("preprocess", self.data[idx]["wav_path"])
        mel_path = os.path.join("preprocess", self.data[idx]["mel_path"])

        output_data["inputs"] = process_mert_format(audio_path)
        output_data["mel"] = preprocess_mel(mel_path, self.target_seq_len)
        return output_data


class ReconstrcutDatasetENCODEC(Data.Dataset):
    def __init__(
        self,
        manifest_path,
        mel_frame_rate,
        input_sec,
    ) -> None:
        super().__init__()

        with open(manifest_path) as f:
            self.data = [json.loads(line) for line in f]

        self.target_seq_len = int(mel_frame_rate * input_sec)
        self.manifest_path = manifest_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return:
            output_data: dict
                {
                    "audio": dict,
                    "mel": torch.tensor
                }
        """
        output_data = dict()

        audio_path = os.path.join("preprocess", self.data[idx]["wav_path"])
        mel_path = os.path.join("preprocess", self.data[idx]["mel_path"])

        output_data["audio"] = preprocess_encodec_format(audio_path)
        output_data["mel"] = preprocess_mel(mel_path, self.target_seq_len)
        return output_data
