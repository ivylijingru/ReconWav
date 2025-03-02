import os
import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from tqdm import tqdm
from typing import List


def process_audio_file(
    input_path: str, output_dir: str, model: EncodecModel, device: torch.device
):
    """处理单个音频文件"""
    try:
        # 加载音频
        wav, sr = torchaudio.load(input_path)
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        wav = wav.unsqueeze(0).to(device)

        # 编码处理
        with torch.no_grad():
            encoded_frames = model.encode(wav)

        # 提取第一个codebook
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        codebook0 = codes[0, 0].cpu().numpy()

        # 生成输出路径
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_codebook0.npy")

        # 保存结果
        np.save(output_path, codebook0)
        return True
    except Exception as e:
        print(f"处理文件 {input_path} 时出错: {str(e)}")
        return False


def find_audio_files(input_dir: str) -> List[str]:
    """查找目录下的所有音频文件"""
    valid_extensions = [".wav", ".mp3", ".flac", ".ogg", ".aac"]
    return [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]


def batch_process(input_dir: str, output_dir: str):
    """批量处理目录下的所有音频文件"""
    # 初始化模型
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取文件列表
    audio_files = find_audio_files(input_dir)
    if not audio_files:
        print(f"在目录 {input_dir} 中未找到支持的音频文件")
        return

    # 处理文件
    success_count = 0
    for file_path in tqdm(audio_files, desc="Processing files"):
        if process_audio_file(file_path, output_dir, model, device):
            success_count += 1

    # 清理资源
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n处理完成！成功处理 {success_count}/{len(audio_files)} 个文件")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量提取Encodec第一个codebook")
    parser.add_argument("--input_dir", type=str, required=True, help="输入音频目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出npy目录")

    args = parser.parse_args()

    batch_process(input_dir=args.input_dir, output_dir=args.output_dir)
