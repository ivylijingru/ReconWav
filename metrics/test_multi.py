import os
import sys
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../../AQUA-Tk"))
sys.path.append(os.path.abspath("../../AQUA-Tk/aquatk/metrics/"))
sys.path.append(os.path.abspath("../../AQUA-Tk/aquatk/metrics/PEAQ"))

import numpy as np
import matplotlib.pyplot as plt

from aquatk.metrics.peaq_metric import process_audio_files
from multiprocessing import Pool, cpu_count


def plot_avg_odg_scores(results, save_path="avg_ODG_scores_error_bars_multi.png"):
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


def process_single_audio_pair(args):
    ref_path, pred_path = args
    _, avg_ODG = process_audio_files(ref_path, pred_path)
    return avg_ODG


def evaluate_audio_quality_parallel(ref_dir, encodec_dir, hifigan_dir, recon_dir):
    """
    使用多进程计算每种音频文件的 avg_ODG。

    Args:
        ref_dir (str): 参考音频文件的目录。
        encodec_dir (str): encodec 处理音频文件的目录。
        hifigan_dir (str): hifigan 处理音频文件的目录。
        recon_dir (str): recon 处理音频文件的目录。

    Returns:
        dict: 包含 encodec、hifigan 和 recon 三种音频文件的 max、min 和 mean avg_ODG 统计结果。
    """
    peaq_encodec = []
    peaq_hifigan = []
    peaq_recon = []

    # 准备参数列表
    encodec_args = []
    hifigan_args = []
    recon_args = []
    
    for filename in os.listdir(ref_dir):
        ref_path = os.path.join(ref_dir, filename)
        encodec_path = os.path.join(encodec_dir, filename)
        hifigan_path = os.path.join(hifigan_dir, filename)
        recon_path = os.path.join(recon_dir, filename)

        # 将文件路径对添加到参数列表
        encodec_args.append((ref_path, encodec_path))
        hifigan_args.append((ref_path, hifigan_path))
        recon_args.append((ref_path, recon_path))

    # 使用多进程计算 avg_ODG
    with Pool(processes=cpu_count()) as pool:
        peaq_encodec = pool.map(process_single_audio_pair, encodec_args)
        peaq_hifigan = pool.map(process_single_audio_pair, hifigan_args)
        peaq_recon = pool.map(process_single_audio_pair, recon_args)

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
    ref_dir = "../../nsynth-valid/audio/"
    encodec_dir = "../samples/encodec"
    hifigan_dir = "../samples/hifigan"
    recon_dir = "../samples/recon"

    result = evaluate_audio_quality_parallel(ref_dir, encodec_dir, hifigan_dir, recon_dir)
