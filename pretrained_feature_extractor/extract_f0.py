import numpy as np
import pyworld

import os
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T


def load_audio(audio_path):
    """Load audio file and convert to mono if necessary."""
    audio, sampling_rate = sf.read(audio_path)
    if audio.ndim > 1:  # Convert to mono if stereo
        audio = audio.mean(axis=1)
    return torch.from_numpy(audio).float(), sampling_rate

def resample_audio(audio, original_rate, target_rate):
    """Resample audio to the target sampling rate."""
    if original_rate != target_rate:
        resampler = T.Resample(original_rate, target_rate)
        audio = resampler(audio.unsqueeze(0)).squeeze(0)  # 添加和移除批次维度以适配torchaudio Resample
    return audio

def extract_f0(audio_path, f0_predictor):
    """
    Extract F0 for a single audio file, resample if necessary, and save it to an output folder.

    Args:
        audio_path (str): Path to the audio file.
        output_folder (str): Path to the folder where the F0 .npy file will be saved.
        f0_predictor (DioF0Predictor): An instance of the DioF0Predictor class.

    Returns:
        None
    """

    # Load and resample audio
    audio, sr = load_audio(audio_path)
    audio = resample_audio(audio, sr, f0_predictor.sampling_rate).numpy()  # Resample to f0_predictor's sample rate

    # Compute F0
    f0_values = f0_predictor.compute_f0(audio)

    return f0_values


class DioF0Predictor:
    def __init__(self, hop_length=512, f0_min=50, f0_max=1100, sampling_rate=44100):
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate
        self.name = "dio"

    def interpolate_f0(self, f0):
        """
        对F0进行插值处理
        """
        vuv_vector = np.zeros_like(f0, dtype=np.float32)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0

        nzindex = np.nonzero(f0)[0]
        data = f0[nzindex]
        nzindex = nzindex.astype(np.float32)
        time_org = self.hop_length / self.sampling_rate * nzindex
        time_frame = np.arange(f0.shape[0]) * self.hop_length / self.sampling_rate

        if data.shape[0] <= 0:
            return np.zeros(f0.shape[0], dtype=np.float32), vuv_vector

        if data.shape[0] == 1:
            return np.ones(f0.shape[0], dtype=np.float32) * f0[0], vuv_vector

        f0 = np.interp(time_frame, time_org, data, left=data[0], right=data[-1])
        
        return f0, vuv_vector

    def resize_f0(self, x, target_len):
        source = np.array(x)
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * target_len, len(source)) / target_len,
            np.arange(0, len(source)),
            source
        )
        res = np.nan_to_num(target)
        return res

    def compute_f0(self, wav, p_len=None):
        if p_len is None:
            p_len = wav.shape[0] // self.hop_length
        f0, t = pyworld.dio(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        return self.interpolate_f0(self.resize_f0(f0, p_len))[0]

    def compute_f0_uv(self, wav, p_len=None):
        if p_len is None:
            p_len = wav.shape[0] // self.hop_length
        f0, t = pyworld.dio(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        return self.interpolate_f0(self.resize_f0(f0, p_len))


# 示例调用
if __name__ == "__main__":
    input_audio = "/path/to/your/audio.wav"
    output_folder = "/path/to/your/output_folder"
    f0_predictor = DioF0Predictor(hop_length=512, f0_min=50, f0_max=1100, sampling_rate=44100)

    # 提取并保存F0
    extract_f0(input_audio, f0_predictor)

