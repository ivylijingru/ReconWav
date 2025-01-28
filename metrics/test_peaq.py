import os
import sys
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../../AQUA-Tk"))
sys.path.append(os.path.abspath("../../AQUA-Tk/aquatk/metrics/"))
sys.path.append(os.path.abspath("../../AQUA-Tk/aquatk/metrics/PEAQ"))

import numpy as np
import matplotlib.pyplot as plt

from aquatk.metrics.peaq_metric import process_audio_files


def plot_avg_odg_scores(results, save_path="avg_ODG_scores_error_bars.png"):
    """
    绘制每种音频类型的 avg_ODG 分数误差棒图。

    Args:
        results (dict): evaluate_audio_quality 函数的返回结果，包括每种音频类型的 max、min 和 mean avg_ODG。
        save_path (str): 图表保存路径，默认为 "avg_ODG_scores_error_bars.png"。
    """
    # 提取 mean、min 和 max 值
    mean_values = [results["encodec"]["mean"], results["hifigan"]["mean"], results["recon"]["mean"]]
    min_values = [results["encodec"]["min"], results["hifigan"]["min"], results["recon"]["min"]]
    max_values = [results["encodec"]["max"], results["hifigan"]["max"], results["recon"]["max"]]

    # 计算误差范围
    yerr = [
        [mean - min_val for mean, min_val in zip(mean_values, min_values)],  # 下误差
        [max_val - mean for mean, max_val in zip(mean_values, max_values)]   # 上误差
    ]

    # 绘制误差棒图
    labels = ['Encodec', 'HiFi GAN', 'Reconstructed']
    x = np.arange(len(labels))

    fig, ax = plt.subplots()
    ax.errorbar(x, mean_values, yerr=yerr, fmt='o', capsize=5, color='royalblue', 
                ecolor='gray', elinewidth=2, markeredgewidth=2)

    # 添加标签和标题
    ax.set_xlabel('Audio Type')
    ax.set_ylabel('avg_ODG Score')
    ax.set_title('avg_ODG Scores with Min and Max Error Bars')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # 保存和显示图表
    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")


def evaluate_audio_quality(ref_dir, encodec_dir, hifigan_dir, recon_dir):
    """
    Evaluate audio quality of three sets of predicted audio files against reference audio files.

    Args:
        ref_dir (str): Directory containing reference audio files.
        encodec_dir (str): Directory containing encodec processed audio files.
        hifigan_dir (str): Directory containing hifigan processed audio files.
        recon_dir (str): Directory containing recon processed audio files.

    Returns:
        dict: Dictionary with max, min, and mean avg_ODG values for each audio set.
    """
    peaq_encodec = []
    peaq_hifigan = []
    peaq_recon = []

    for filename in os.listdir(ref_dir):
        ref_path = os.path.join(ref_dir, filename)
        encodec_path = os.path.join(encodec_dir, filename)
        hifigan_path = os.path.join(hifigan_dir, filename)
        recon_path = os.path.join(recon_dir, filename)

        # 计算 encodec 的 avg_ODG
        _, avg_ODG_encodec = process_audio_files(ref_path, encodec_path)
        peaq_encodec.append(avg_ODG_encodec)

        # 计算 hifigan 的 avg_ODG
        _, avg_ODG_hifigan = process_audio_files(ref_path, hifigan_path)
        peaq_hifigan.append(avg_ODG_hifigan)

        # 计算 recon 的 avg_ODG
        _, avg_ODG_recon = process_audio_files(ref_path, recon_path)
        peaq_recon.append(avg_ODG_recon)

    # 计算三个数组的最大值、最小值和平均值
    result = {
        "encodec": {
            "max": max(peaq_encodec),
            "min": min(peaq_encodec),
            "mean": sum(peaq_encodec) / len(peaq_encodec) if peaq_encodec else 0
        },
        "hifigan": {
            "max": max(peaq_hifigan),
            "min": min(peaq_hifigan),
            "mean": sum(peaq_hifigan) / len(peaq_hifigan) if peaq_hifigan else 0
        },
        "recon": {
            "max": max(peaq_recon),
            "min": min(peaq_recon),
            "mean": sum(peaq_recon) / len(peaq_recon) if peaq_recon else 0
        }
    }

    return result


if __name__ == "__main__":
    # ref_dir = "../../nsynth-valid/audio/"
    # encodec_dir = "../samples/encodec"
    # hifigan_dir = "../samples/hifigan"
    # recon_dir = "../samples/recon"

    # result = evaluate_audio_quality(ref_dir, encodec_dir, hifigan_dir, recon_dir)

    # Encodec Results:
    #      max: -1.1070
    #      min: -3.9130
    #     mean: -3.2649

    # Hifigan Results:
    #      max: -0.5530
    #      min: -3.8930
    #     mean: -2.9484

    # Recon Results:
    #      max: -0.6420
    #      min: -3.9130
    #     mean: -2.9848

    result = {
        "encodec": {
            "max": -1.1070,
            "min": -3.9130,
            "mean": -3.2649
        },
        "hifigan": {
            "max": -0.5530,
            "min": -3.8930,
            "mean": -2.9484
        },
        "recon": {
            "max": -0.6420,
            "min": -3.9130,
            "mean": -2.9848
        }
    }

    plot_avg_odg_scores(result, save_path="100-ODG.png")
