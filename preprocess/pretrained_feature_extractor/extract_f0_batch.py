import os
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
import pyworld
from multiprocessing import Pool, cpu_count


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

def extract_f0_single_file(args):
    """
    Helper function to extract F0 for a single audio file (for multiprocessing).
    Args:
        args (tuple): Contains audio_path and f0_predictor.
    Returns:
        tuple: (audio_path, f0_values)
    """
    audio_path, f0_predictor = args

    # Load and resample audio
    audio, sr = load_audio(audio_path)
    audio = resample_audio(audio, sr, f0_predictor.sampling_rate).numpy()  # Resample to f0_predictor's sample rate

    # Compute F0
    f0_values = f0_predictor.compute_f0(audio)

    return audio_path, f0_values

def batch_extract_f0(audio_files, f0_predictor, output_folder):
    """
    Batch extract F0 for multiple audio files using multiprocessing.

    Args:
        audio_files (list): List of paths to audio files.
        f0_predictor (DioF0Predictor): An instance of the DioF0Predictor class.
        output_folder (str): Path to the folder where F0 .npy files will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

    # Prepare arguments for each audio file
    args = [(audio_path, f0_predictor) for audio_path in audio_files]

    # Use multiprocessing Pool to process files in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(extract_f0_single_file, args)

    # Save results
    for audio_path, f0_values in results:
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        save_path = os.path.join(output_folder, f"{filename}_f0.npy")
        np.save(save_path, f0_values)
        print(f"F0 extracted and saved to {save_path}")

# 示例调用
if __name__ == "__main__":
    input_folder = "../../../nsynth-valid/audio/"
    output_folder = "../../samples-f0/output_folder"
    f0_predictor = DioF0Predictor(hop_length=512, f0_min=50, f0_max=1100, sampling_rate=44100)

    # 获取所有音频文件路径
    audio_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.wav', '.mp3'))]

    # 批量提取F0
    batch_extract_f0(audio_files, f0_predictor, output_folder)
