import json
import os
import numpy as np
import soundfile as sf
import torchaudio.transforms as T

import torch
import torch.utils.data as Data
from transformers import Wav2Vec2FeatureExtractor
import soundfile as sf
import torchaudio.transforms as T


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
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v0-public",trust_remote_code=True)
        self.manifest_path = manifest_path

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        mel
        vgg
        """
        output_data = dict()
        # load audio
        audio_path = os.path.join("preprocess", self.data[idx]["wav_path"])
        audio, sampling_rate = sf.read(audio_path)

        # convert to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio_array = torch.from_numpy(audio).float()

        # resample
        resample_rate = self.processor.sampling_rate
        if resample_rate != sampling_rate:
            resampler = T.Resample(sampling_rate, resample_rate)
        else:
            resampler = None
        if resampler is None:
            input_audio = audio_array
        else:
            input_audio = resampler(audio_array)
        
        # process and extract embeddings
        inputs = self.processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")
        
        for input_key in inputs.keys():
            inputs[input_key] = inputs[input_key].squeeze(0)

        output_data["inputs"] = inputs

        if "train" in self.manifest_path:
            output_folder = "preprocess/encodec_feature_train"
        else:
            output_folder = "preprocess/encodec_feature_valid"
        part_name = os.path.splitext(os.path.basename(audio_path))[0]
        encodec_path = os.path.join(output_folder, f"{part_name}_encodec.npy")
        encodec = torch.from_numpy(np.load(encodec_path))
        output_data["encodec"] = encodec

        # output_data = dict()
        # mel = torch.from_numpy(np.load(os.path.join("preprocess", self.data[idx]["mel_path"]))).float()
        # output_data["mel"] = torch.zeros(mel.shape[0], self.target_seq_len)
        # if mel.shape[1] < self.target_seq_len:
        #     output_data["mel"][:, :mel.shape[1]] = mel
        # elif mel.shape[1] >= self.target_seq_len:
        #     output_data["mel"] = mel[:, :self.target_seq_len]

        # if encodec_feat.shape[2] < self.encodec_seq_len:
        #     output_data["audio"][:, :, :encodec_feat.shape[2]] = encodec_feat
        # elif encodec_feat.shape[2] >= self.encodec_seq_len:
        #     output_data["audio"] = encodec_feat[:, :, :self.encodec_seq_len]
        # output_data["audio"] = output_data["audio"].squeeze()

        # print(output_data["encodec"].shape) [1, 128, 300]

        return output_data

