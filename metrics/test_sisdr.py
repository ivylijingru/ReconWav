import os
import sys
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../../AQUA-Tk"))
sys.path.append(os.path.abspath("../../AQUA-Tk/aquatk/metrics/"))
sys.path.append(os.path.abspath("../../AQUA-Tk/aquatk/metrics/PEAQ"))

import soundfile as sf
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchaudio.transforms as T

from aquatk.metrics.peaq_metric import process_audio_files
from aquatk.metrics.errors import si_sdr

from tqdm import tqdm


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


def plot_avg_SI_SDR_scores(results, save_path="avg_SISDR_scores_error_bars.png"):
    """
    绘制每种音频类型的 avg_SI_SDR 分数误差棒图。

    Args:
        results (dict): evaluate_audio_quality 函数的返回结果，包括每种音频类型的 max、min 和 mean avg_SI_SDR。
        save_path (str): 图表保存路径，默认为 "avg_SI_SDR_scores_error_bars.png"。
    """
    # 提取 mean、min 和 max 值
    mean_values = [results["encodec"]["mean"], results["hifigan"]["mean"], results["recon"]["mean"], results["recon-hifigan"]["mean"]]
    min_values = [results["encodec"]["min"], results["hifigan"]["min"], results["recon"]["min"], results["recon-hifigan"]["min"]]
    max_values = [results["encodec"]["max"], results["hifigan"]["max"], results["recon"]["max"], results["recon-hifigan"]["max"]]

    # 计算误差范围
    yerr = [
        [mean - min_val for mean, min_val in zip(mean_values, min_values)],  # 下误差
        [max_val - mean for mean, max_val in zip(mean_values, max_values)]   # 上误差
    ]

    # 绘制误差棒图
    labels = ['Encodec', 'HiFi GAN', 'Reconstructed', 'Recon-Hifi']
    x = np.arange(len(labels))

    fig, ax = plt.subplots()
    ax.errorbar(x, mean_values, yerr=yerr, fmt='o', capsize=5, color='royalblue', 
                ecolor='gray', elinewidth=2, markeredgewidth=2)

    # 添加标签和标题
    ax.set_xlabel('Audio Type')
    ax.set_ylabel('avg_SI_SDR Score')
    ax.set_title('avg_SI_SDR Scores with Min and Max Error Bars')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # 保存和显示图表
    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")


def calculate_sisdr(ref_path, pred_path):
    ref_arr, ref_sr = load_audio(ref_path)
    pred_arr, pred_sr = load_audio(pred_path)
    pred_arr = resample_audio(pred_arr, pred_sr, ref_sr)
    min_length = min(pred_arr.shape[0], ref_arr.shape[0])
    ref_arr = ref_arr[:min_length]
    pred_arr = pred_arr[:min_length]
    res_si_sdr = si_sdr(ref_arr.numpy(), pred_arr.numpy())
    return res_si_sdr


def plot_pitch_vs_si_sdr(pitch, si_sdr, save_path="pitch_vs_si_sdr.png"):
    """
    绘制 Pitch 与 SI_SDR 的二维散点图，并保存到磁盘。

    Args:
        pitch (list or array): 横轴数据，表示 pitch 值。
        si_sdr (list or array): 纵轴数据，表示 SI_SDR 值。
        save_path (str): 图像保存路径，默认为 "pitch_vs_si_sdr.png"。
    """
    plt.figure(figsize=(8, 6))
    
    # 绘制散点图
    plt.scatter(pitch, si_sdr, color='royalblue', alpha=0.6, edgecolor='k', s=50)
    
    # 添加标题和标签
    plt.title("Pitch vs. SI_SDR")
    plt.xlabel("Pitch")
    plt.ylabel("SI_SDR (dB)")
    
    # 保存图像到指定路径
    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"图像已保存到 {save_path}")


def evaluate_audio_quality(ref_dir, encodec_dir, hifigan_dir, recon_dir):
    """
    Evaluate audio quality of three sets of predicted audio files against reference audio files.

    Args:
        ref_dir (str): Directory containing reference audio files.
        encodec_dir (str): Directory containing encodec processed audio files.
        hifigan_dir (str): Directory containing hifigan processed audio files.
        recon_dir (str): Directory containing recon processed audio files.

    Returns:
        dict: Dictionary with max, min, and mean avg_SI_SDR values for each audio set.
    """
    sisdr_encodec_list = []
    sisdr_hifigan_list = []
    sisdr_recon_list = []
    sisdr_recon_hifigan_list = []

    timbre_pitch_dict = {}

    for filename in tqdm(os.listdir(ref_dir)):
        ref_path = os.path.join(ref_dir, filename)
        encodec_path = os.path.join(encodec_dir, filename)
        hifigan_path = os.path.join(hifigan_dir, filename)
        recon_path = os.path.join(recon_dir, filename)

        if not os.path.exists(encodec_path):
            break

        # 计算 encodec 的 avg_SI_SDR
        sisdr_encodec = calculate_sisdr(ref_path, encodec_path)
        sisdr_encodec_list.append(sisdr_encodec)

        # 计算 hifigan 的 avg_SI_SDR
        sisdr_hifigan = calculate_sisdr(ref_path, hifigan_path)
        sisdr_hifigan_list.append(sisdr_hifigan)

        # 计算 recon 的 avg_SI_SDR
        sisdr_recon = calculate_sisdr(ref_path, recon_path)
        sisdr_recon_list.append(sisdr_recon)

        # 计算 recon 对 hifigan
        sisdr_recon_hifigan = calculate_sisdr(hifigan_path, recon_path)
        sisdr_recon_hifigan_list.append(sisdr_recon_hifigan)

        instrument_id = filename.split("-")[0]
        if instrument_id not in timbre_pitch_dict.keys():
            timbre_pitch_dict[instrument_id] = []
        timbre_pitch_dict[instrument_id].append([filename.split("-")[1], sisdr_recon])

    for key in timbre_pitch_dict.keys():
        pitch = [int(sub[0].lstrip('0')) for sub in timbre_pitch_dict[key]]
        sisdr = [int(sub[1]) for sub in timbre_pitch_dict[key]]
        plot_pitch_vs_si_sdr(pitch, sisdr, save_path=f"{key}.png")

    # 计算三个数组的最大值、最小值和平均值
    result = {
        "encodec": {
            "max": max(sisdr_encodec_list),
            "min": min(sisdr_encodec_list),
            "mean": sum(sisdr_encodec_list) / len(sisdr_encodec_list) if sisdr_encodec_list else 0
        },
        "hifigan": {
            "max": max(sisdr_hifigan_list),
            "min": min(sisdr_hifigan_list),
            "mean": sum(sisdr_hifigan_list) / len(sisdr_hifigan_list) if sisdr_hifigan_list else 0
        },
        "recon": {
            "max": max(sisdr_recon_list),
            "min": min(sisdr_recon_list),
            "mean": sum(sisdr_recon_list) / len(sisdr_recon_list) if sisdr_recon_list else 0
        },
        "recon-hifigan": {
            "max": max(sisdr_recon_hifigan_list),
            "min": min(sisdr_recon_hifigan_list),
            "mean": sum(sisdr_recon_hifigan_list) / len(sisdr_recon_hifigan_list) if sisdr_recon_hifigan_list else 0
        },
    }

    return result


if __name__ == "__main__":
    ref_dir = "../../nsynth-valid/audio/"
    encodec_dir = "../samples-new-hifi/encodec"
    hifigan_dir = "../samples-new-hifi/hifigan"
    recon_dir = "../samples-new-hifi/recon"

    result = evaluate_audio_quality(ref_dir, encodec_dir, hifigan_dir, recon_dir)
    plot_avg_SI_SDR_scores(result, save_path="avg_SISDR_scores_error_bars--new-hifi.png")
