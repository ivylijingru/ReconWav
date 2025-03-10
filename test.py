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
import soundfile as sf
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from hifigan.models import Generator
import torchaudio.transforms as T

# 保存音频数据为 .wav 文件
# sf.write("output.wav", audio_data, sample_rate)

from encodec import EncodecModel
encoder = EncodecModel.encodec_model_24khz(pretrained=True).encoder
decoder = EncodecModel.encodec_model_24khz(pretrained=True).decoder

ENCODEC_SR = 24000
PESQ_SR = 16000
HIFIGAN_SR = 22050


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
    config_file = "../hifigan/checkpoint/config.json"
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    device = torch.device('cpu')
    generator = Generator(h).to(device)

    checkpoint_file = "../hifigan/checkpoint/generator_v1"
    state_dict_g = load_checkpoint(checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()

    mel_module.h = h

    pesq_encodec = []
    pesq_hifigan = []
    pesq_recon = []

    cnt = 0
    for audio_file in os.listdir(valid_audio_dir):
        filename = os.path.join(valid_audio_dir, audio_file)

        wav_encodec, sr = load_wav(filename)

        # For encodec sanity check
        number_of_samples = round(len(wav_encodec) * float(ENCODEC_SR) / sr)
        wav_encodec = sps.resample(wav_encodec, number_of_samples)
        wav_encodec = torch.FloatTensor(wav_encodec)
        normalize_max = wav_encodec.abs().max()
        wav_encodec = wav_encodec / normalize_max
        wav_encodec = wav_encodec.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            encoded_frames = encoder(wav_encodec)
            output = decoder(encoded_frames)
        encodec_sanity_audio = output.detach().squeeze()

        pesq = PerceptualEvaluationSpeechQuality(PESQ_SR, "wb")
        origin_wav = resample_wave(wav_encodec.squeeze(0), original_rate=ENCODEC_SR, target_rate=PESQ_SR)
        encodec_wav = resample_wave(encodec_sanity_audio.unsqueeze(0), original_rate=ENCODEC_SR, target_rate=PESQ_SR)
        # sf.write(f"origin_{cnt}.wav", origin_wav.squeeze().numpy(), PESQ_SR)
        # sf.write(f"encodec_{cnt}.wav", encodec_wav.squeeze().numpy(), PESQ_SR)
        try:
            pesq_score_encodec = pesq(encodec_wav, origin_wav).item()
        except Exception as e:
            print("An unexpected error occurred:", e)
            pesq_score_encodec = -1

        print("PESQ Score -- Encodec sanity check: ", pesq_score_encodec)
        if pesq_score_encodec != -1:
            pesq_encodec.append(pesq_score_encodec)

        wav, sr = load_wav(filename)
        number_of_samples = round(len(wav) * float(HIFIGAN_SR) / sr)
        wav = sps.resample(wav, number_of_samples)
        wav = wav / MAX_WAV_VALUE
        wav = torch.FloatTensor(wav)
        x = get_mel(wav.unsqueeze(0))

        with torch.no_grad():
            y_g_hat = generator(x)
            hifi_audio = y_g_hat.squeeze()
            # hifi_audio = hifi_audio * MAX_WAV_VALUE
            hifi_audio = hifi_audio.cpu().unsqueeze(0)

        origin_wav = resample_wave(wav.unsqueeze(0), original_rate=HIFIGAN_SR, target_rate=PESQ_SR)
        hifi_wav = resample_wave(hifi_audio, original_rate=HIFIGAN_SR, target_rate=PESQ_SR)
        min_length = min(hifi_wav.shape[1], origin_wav.shape[1])
        hifi_wav = hifi_wav[:, :min_length]
        origin_wav = origin_wav[:, :min_length]
        try:
            pesq_score_hifigan = pesq(hifi_wav, origin_wav).item()
        except Exception as e:
            print("An unexpected error occurred:", e)
            pesq_score_hifigan = -1

        print("PESQ Score -- HIFI GAN: ", pesq_score_hifigan)
        if pesq_score_hifigan != -1:
            pesq_hifigan.append(pesq_score_hifigan)
        sf.write(f"hifigan_{cnt}.wav", encodec_wav.squeeze().numpy(), PESQ_SR)

        # Reconstruct model
        from models import ReconstructModel
        checkpoint_path = "/home/jli3268/ReconWav/work_dir_encodec/weight_v6/epoch=29-val_loss-total=0.917.ckpt"
        model = ReconstructModel.load_from_checkpoint(checkpoint_path)

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

        audio_torch, sr = load_audio(filename)
        audio_resampled = resample_audio(audio_torch, sr, ENCODEC_SR)
        audio_resampled = audio_resampled.unsqueeze(0).unsqueeze(0)

        mel_output = model.inference_step(audio_resampled)

        with torch.no_grad():
            y_g_hat = generator(mel_output)
            recon_audio = y_g_hat.squeeze()
            recon_audio = recon_audio.cpu()

        origin_wav = resample_wave(wav.unsqueeze(0), original_rate=HIFIGAN_SR, target_rate=PESQ_SR)
        recon_audio = resample_wave(recon_audio.unsqueeze(0), original_rate=HIFIGAN_SR, target_rate=PESQ_SR)
        min_length = min(recon_audio.shape[1], origin_wav.shape[1])
        recon_audio = recon_audio[:, :min_length]
        origin_wav = origin_wav[:, :min_length]

        try:
            pesq_score_recon = pesq(recon_audio, origin_wav).item()
        except Exception as e:
            print("An unexpected error occurred:", e)
            pesq_score_recon = -1

        print("PESQ Score -- Reconstructed diff: ", pesq_score_recon)
        if pesq_score_recon != -1:
            pesq_recon.append(pesq_score_recon)
        sf.write(f"recon_diff_{cnt}.wav", recon_audio.squeeze().numpy(), PESQ_SR)
        cnt += 1
        if cnt == 10:
            break

    import numpy as np
    import matplotlib.pyplot as plt

    # 计算每种 PESQ 评分的平均值、最小值和最大值
    mean_values = [np.mean(pesq_encodec), np.mean(pesq_hifigan), np.mean(pesq_recon)]
    min_values = [min(pesq_encodec), min(pesq_hifigan), min(pesq_recon)]
    max_values = [max(pesq_encodec), max(pesq_hifigan), max(pesq_recon)]

    # 计算误差范围
    yerr = [ [mean - min_val for mean, min_val in zip(mean_values, min_values)],  # 下误差
            [max_val - mean for mean, max_val in zip(mean_values, max_values)] ]  # 上误差

    # 绘制误差棒图
    labels = ['Encodec', 'HiFi GAN', 'Reconstructed']
    x = np.arange(len(labels))

    fig, ax = plt.subplots()
    ax.errorbar(x, mean_values, yerr=yerr, fmt='o', capsize=5, color='royalblue', ecolor='gray', elinewidth=2, markeredgewidth=2)

    # 添加标签和标题
    ax.set_xlabel('PESQ Type')
    ax.set_ylabel('PESQ Score')
    ax.set_title('PESQ Scores with Min and Max Error Bars')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # 保存图表
    plt.savefig("pesq_scores_error_bars_tmux.png", format="png", dpi=300, bbox_inches="tight")
