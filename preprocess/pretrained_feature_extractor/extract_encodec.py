import os
import numpy as np
from encodec import EncodecModel
import torch
import soundfile as sf
import torchaudio.transforms as T

ENCODEC_SR = 24000
DEVICE = torch.device("cuda")

# 初始化 encoder
encoder = EncodecModel.encodec_model_24khz(pretrained=True).encoder.cuda()


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
        audio = resampler(audio)
    return audio


def batch_extract_encodec(file_paths, output_folder, encoder, batch_size=64):
    """
    Batch extract encodec features for multiple audio files and save them to an output folder.

    Args:
        file_paths (list): List of paths to audio files.
        output_folder (str): Path to the folder where encoded .npy files will be saved.
        encoder (torch.nn.Module): The encoder model.
        batch_size (int): Number of files to process in each batch.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在

    # 将文件按批次处理
    for i in range(0, len(file_paths), batch_size):
        batch_files = file_paths[i : i + batch_size]
        audio_tensors = []

        for file_path in batch_files:
            # 加载并预处理音频
            audio, sr = load_audio(file_path)
            audio = resample_audio(audio, sr, ENCODEC_SR)
            audio = audio / audio.abs().max()  # 归一化
            audio = audio.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度
            audio_tensors.append(audio)

        # 将音频批次堆叠，并移动到 GPU
        batch_audio = torch.cat(audio_tensors, dim=0).to(DEVICE)

        # 批量编码
        with torch.no_grad():
            encoded_frames = encoder(batch_audio)

        # 将编码结果分割并保存
        for j, file_path in enumerate(batch_files):
            filename = os.path.splitext(os.path.basename(file_path))[0]
            save_path = os.path.join(output_folder, f"{filename}_encodec.npy")
            np.save(save_path, encoded_frames[j].cpu().numpy())
            print(f"Encoded frames saved to {save_path}")


def extract_multiple_encodec_feature(input_dir, output_dir, fs=44100):
    file_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith((".wav", ".mp3"))
    ]
    # file_paths = file_paths[:2] # HACK!!!
    batch_extract_encodec(file_paths, output_dir, encoder, batch_size=64)


if __name__ == "__main__":
    input_folder = "/home/jli3268/nsynth-train/audio"
    output_folder = "../encodec_feature_train"
    file_paths = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith((".wav", ".mp3"))
    ]

    # 调用 batch_extract_encodec 处理文件
    batch_extract_encodec(file_paths, output_folder, encoder, batch_size=4)
