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

from transformers import Wav2Vec2FeatureExtractor

# instantiate the model. You can optionally set use_cuda_kernel=True for faster inference.
model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_256x', use_cuda_kernel=False)

# remove weight norm in the model and set to eval mode
model.remove_weight_norm()
model = model.eval().to(device)

def create_if_not_exist(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

output_dir = "samples-bigvgan-mert-128-nonan"
recon_output_dir = os.path.join(output_dir, "recon")
valid_audio_dir = "../nsynth-valid/audio"
create_if_not_exist(recon_output_dir)

from models import ReconstructModel
# checkpoint_path = "work_dir_mert/weight_new_data_128bands/epoch=0-val_loss-total=1.226.ckpt"
checkpoint_path = "work_dir_mert/weight_new_data_128bands_1e-6/epoch=57-val_loss-total=1.041.ckpt"
recon_model = ReconstructModel.load_from_checkpoint(checkpoint_path).to(device)

processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v0-public",trust_remote_code=True)
resample_rate = processor.sampling_rate

for name, param in recon_model.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN found in parameter: {name}")

for audio_file in tqdm(os.listdir(valid_audio_dir)):
    # load wav file and compute mel spectrogram
    wav_path = os.path.join(valid_audio_dir, audio_file)
    
    wav, sr = librosa.load(wav_path, sr=resample_rate, mono=True) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
    wav = torch.FloatTensor(wav).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]
    mel_output = recon_model.inference_step(wav)

    # generate waveform from mel
    with torch.inference_mode():
        wav_gen = model(mel_output) # wav_gen is FloatTensor with shape [B(1), 1, T_time] and values in [-1, 1]
    wav_gen_float = wav_gen.squeeze(0).cpu() # wav_gen is FloatTensor with shape [1, T_time]

    part_name = os.path.splitext(os.path.basename(wav_path))[0]
    # you can convert the generated waveform to 16 bit linear PCM
    sf.write(os.path.join(recon_output_dir, f"{part_name}.wav"), wav_gen_float.squeeze().numpy(), model.h.sampling_rate)