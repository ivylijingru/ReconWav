import os
import sys
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../hifigan"))
from hifigan.inference_mel import get_mel, load_wav, MAX_WAV_VALUE
import hifigan.inference_mel as mel_module
import scipy.signal as sps
import json
import torch
import torchaudio
import numpy as np
import soundfile as sf
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from hifigan.models import Generator
import torchaudio.transforms as T
from time import time
from tqdm import tqdm

device = torch.device('cuda')

from encodec import EncodecModel
encoder = EncodecModel.encodec_model_24khz(pretrained=True).encoder
decoder = EncodecModel.encodec_model_24khz(pretrained=True).decoder

ENCODEC_SR = 24000
PESQ_SR = 16000
HIFIGAN_SR = 22050

# Reconstruct model
from models import ReconstructModel
checkpoint_path = "/home/jli3268/ReconWav/work_dir_encodec/weight_v6/epoch=29-val_loss-total=0.917.ckpt"
model = ReconstructModel.load_from_checkpoint(checkpoint_path).to(device)

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

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def resample_wave(wave_tensor: torch.Tensor, original_rate: int, target_rate: int) -> torch.Tensor:
    """
    对 PyTorch wave tensor 进行重采样。
    
    参数:
    - wave_tensor (torch.Tensor): 音频数据的张量，形状为 (num_channels, num_samples)。
    - original_rate (int): 原始采样率。
    - target_rate (int): 目标采样率。
    
    返回:
    - torch.Tensor: 重采样后的 wave tensor。
    """
    resampler = torchaudio.transforms.Resample(orig_freq=original_rate, new_freq=target_rate)
    return resampler(wave_tensor)


if __name__ == "__main__":
    valid_audio_dir = "../nsynth-valid/audio"

    # For hifigan
    # config_file = "../hifigan/checkpoint/config.json"
    config_file = "UNIVERSAL_V1/config.json"
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = Generator(h).to(device)

    # checkpoint_file = "../hifigan/checkpoint/generator_v1"
    checkpoint_file = "UNIVERSAL_V1/g_02500000"
    state_dict_g = load_checkpoint(checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()
    
    encoder.eval()
    decoder.eval()

    mel_module.h = h

    cnt = 0

    start_time = time()

    output_dir = "samples-new-hifi"
    encodec_output_dir = os.path.join(output_dir, "encodec")
    hifigan_output_dir = os.path.join(output_dir, "hifigan")
    recon_output_dir = os.path.join(output_dir, "recon")

    def create_if_not_exist(filepath):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
    
    create_if_not_exist(output_dir)
    create_if_not_exist(encodec_output_dir)
    create_if_not_exist(hifigan_output_dir)
    create_if_not_exist(recon_output_dir)

    for audio_file in tqdm(os.listdir(valid_audio_dir)):
        filename = os.path.join(valid_audio_dir, audio_file)

        output_folder = "preprocess/encodec_feature_valid"
        part_name = os.path.splitext(os.path.basename(filename))[0]
        audio_resampled = None
        encodec_path = os.path.join(output_folder, f"{part_name}_encodec.npy")
        encodec_arr = torch.from_numpy(np.load(encodec_path)).unsqueeze(0)
        output = decoder(encodec_arr)
        encodec_sanity_audio = output.detach().squeeze().cpu()
        sf.write(os.path.join(encodec_output_dir, f"{part_name}.wav"), encodec_sanity_audio.squeeze().numpy(), ENCODEC_SR)

        wav, sr = load_wav(filename)
        number_of_samples = round(len(wav) * float(HIFIGAN_SR) / sr)
        wav = sps.resample(wav, number_of_samples)
        wav = wav / MAX_WAV_VALUE
        wav = torch.FloatTensor(wav)
        x = get_mel(wav.unsqueeze(0)).to(device)
        part_name = os.path.splitext(os.path.basename(filename))[0]

        with torch.no_grad():
            y_g_hat = generator(x)
            hifi_audio = y_g_hat.squeeze()
            # hifi_audio = hifi_audio * MAX_WAV_VALUE
            hifi_audio = hifi_audio.cpu().unsqueeze(0)

        sf.write(os.path.join(hifigan_output_dir, f"{part_name}.wav"), hifi_audio.squeeze().numpy(), HIFIGAN_SR)

        # audio_torch, sr = load_audio(filename)
        # audio_resampled = resample_audio(audio_torch, sr, ENCODEC_SR)
        # audio_resampled = audio_resampled.unsqueeze(0).unsqueeze(0).to(device)

        output_folder = "preprocess/encodec_feature_valid"
        part_name = os.path.splitext(os.path.basename(filename))[0]
        audio_resampled = None
        encodec_path = os.path.join(output_folder, f"{part_name}_encodec.npy")
        mel_output = model.inference_step(audio_resampled, encodec_path=encodec_path)

        with torch.no_grad():
            y_g_hat = generator(mel_output)
            recon_audio = y_g_hat.squeeze()
            recon_audio = recon_audio.cpu()

        sf.write(os.path.join(recon_output_dir, f"{part_name}.wav"), recon_audio.squeeze().numpy(), HIFIGAN_SR)

    end_time = time()
    print(end_time - start_time)
