device = 'cuda'

import sys
import os

sys.path.append("/home/jli3268/BigVGAN")

import torch
import bigvgan
import librosa
from meldataset import get_mel_spectrogram

import json
import torch
import torchaudio
import numpy as np
import soundfile as sf
import torchaudio.transforms as T
from time import time
from tqdm import tqdm


# instantiate the model. You can optionally set use_cuda_kernel=True for faster inference.
model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_256x', use_cuda_kernel=False)

# remove weight norm in the model and set to eval mode
model.remove_weight_norm()
model = model.eval().to(device)

def create_if_not_exist(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

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

output_dir = "samples-bigvgan-128-debug"
recon_output_dir = os.path.join(output_dir, "recon")
valid_audio_dir = "../nsynth-valid/audio"
create_if_not_exist(recon_output_dir)

from models import ReconstructModel
checkpoint_path = "work_dir_encodec_128bands/weight_new_data/epoch=127-val_loss-total=0.498.ckpt"
recon_model = ReconstructModel.load_from_checkpoint(checkpoint_path).to(device)

for audio_file in tqdm(os.listdir(valid_audio_dir)):
    # load wav file and compute mel spectrogram
    wav_path = os.path.join(valid_audio_dir, audio_file)
    # wav, sr = librosa.load(wav_path, sr=model.h.sampling_rate, mono=True) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
    # wav = torch.FloatTensor(wav).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]

    output_data = dict()
    audio_torch, sr = load_audio(wav_path)
    audio_resampled = resample_audio(audio_torch, sr, 24000)
    audio_resampled = audio_resampled.unsqueeze(0).unsqueeze(0).cuda()
    # output_data["audio"] = audio_resampled
    mel_output1 = recon_model.inference_step(audio_resampled)

    output_folder = "preprocess/encodec_feature_valid"
    part_name = os.path.splitext(os.path.basename(wav_path))[0]
    audio_resampled = None
    encodec_path = os.path.join(output_folder, f"{part_name}_encodec.npy")
    mel_output2 = recon_model.inference_step(audio_resampled, encodec_path=encodec_path)
    print(mel_output1 - mel_output2)
    # generate waveform from mel
    with torch.inference_mode():
        wav_gen = model(mel_output1) # wav_gen is FloatTensor with shape [B(1), 1, T_time] and values in [-1, 1]
    wav_gen_float = wav_gen.squeeze(0).cpu() # wav_gen is FloatTensor with shape [1, T_time]

    # you can convert the generated waveform to 16 bit linear PCM
    sf.write(os.path.join(recon_output_dir, f"{part_name}.wav"), wav_gen_float.squeeze().numpy(), model.h.sampling_rate)
    break