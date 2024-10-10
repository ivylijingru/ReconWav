import json
import os
import numpy as np
import soundfile as sf
import torchaudio.transforms as T

import torch
import torch.utils.data as Data


def load_audio(audio_path):
    """Load audio file and convert to mono if necessary."""
    audio, sampling_rate = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return torch.from_numpy(audio).float(), sampling_rate


def resample_audio(audio_array, original_rate, target_rate):
    """Resample audio to the target sampling rate."""
    if original_rate != target_rate:
        resampler = T.Resample(original_rate, target_rate)
        return resampler(audio_array)
    return audio_array


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

        self.audio_seq_len = 16000 * input_sec
        self.encodec_sr = 24000

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        mel
        vgg
        """
        output_data = dict()
        mel = torch.from_numpy(np.load(os.path.join("preprocess", self.data[idx]["mel_path"]))).float()
        output_data["mel"] = torch.zeros(mel.shape[0], self.target_seq_len)
        if mel.shape[1] < self.target_seq_len:
            output_data["mel"][:, :mel.shape[1]] = mel
        elif mel.shape[1] >= self.target_seq_len:
            output_data["mel"] = mel[:, :self.target_seq_len]

        audio_path = os.path.join("preprocess", self.data[idx]["wav_path"])
        audio_torch, sr = load_audio(audio_path)
        audio_resampled = resample_audio(audio_torch, sr, self.encodec_sr)

        audio_resampled = audio_resampled.unsqueeze(0)
        output_data["audio"] = audio_resampled

        # if encodec_feat.shape[2] < self.encodec_seq_len:
        #     output_data["audio"][:, :, :encodec_feat.shape[2]] = encodec_feat
        # elif encodec_feat.shape[2] >= self.encodec_seq_len:
        #     output_data["audio"] = encodec_feat[:, :, :self.encodec_seq_len]
        # output_data["audio"] = output_data["audio"].squeeze()

        # print(output_data["encodec"].shape) [1, 128, 300]

        return output_data

