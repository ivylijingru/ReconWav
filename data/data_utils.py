import numpy as np
import torch
import soundfile as sf
import torchaudio.transforms as T

from transformers import Wav2Vec2FeatureExtractor

ECODEC_SR = 24000

# 在模块内部缓存 processor（单例模式）
_processor = None


def get_processor():
    global _processor
    if _processor is None:
        _processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v0-public", trust_remote_code=True
        )
    return _processor


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


def process_mert_format(audio_path, is_inference=False):
    processor = get_processor()
    """Process audio file with Wav2Vec2 processor and return the embeddings."""
    audio, sampling_rate = sf.read(audio_path)
    # convert to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    audio_array = torch.from_numpy(audio).float()

    # resample
    resample_rate = processor.sampling_rate
    if resample_rate != sampling_rate:
        resampler = T.Resample(sampling_rate, resample_rate)
    else:
        resampler = None
    if resampler is None:
        input_audio = audio_array
    else:
        input_audio = resampler(audio_array)

    # process and extract embeddings
    inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")

    # squeeze batch dimension
    if not is_inference:
        for input_key in inputs.keys():
            inputs[input_key] = inputs[input_key].squeeze(0)

    return inputs


def preprocess_encodec_format(audio_path):
    """"""
    audio_torch, sr = load_audio(audio_path)
    audio_resampled = resample_audio(audio_torch, sr, ECODEC_SR)

    audio_resampled = audio_resampled.unsqueeze(0)
    return audio_resampled


def preprocess_mel(mel_path, target_seq_len):
    """Preprocess mel spectrogram and pad or truncate to the target sequence length."""
    mel = torch.from_numpy(np.load(mel_path)).float()
    tmp_data = torch.zeros(mel.shape[0], target_seq_len)
    if mel.shape[1] < target_seq_len:
        tmp_data[:, : mel.shape[1]] = mel
    elif mel.shape[1] >= target_seq_len:
        tmp_data = mel[:, :target_seq_len]
    return tmp_data


def preprocess_codebook(cb_path):
    """Preprocess codebook and pad or truncate to the target sequence length."""
    codebook = torch.from_numpy(np.load(cb_path)).long()
    return codebook
