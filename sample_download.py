import os
import random
import shutil
import subprocess

# 文件夹路径设置
base_path = 'samples-new-hifi'  # 替换为你的 samples-new-hifi 文件夹路径
ground_truth_path = '../nsynth-valid/audio/'  # 替换为 ground truth 文件夹路径
output_folder = 'sampled-100'  # 替换为新文件夹路径

# 子文件夹名列表
subfolders = ['encodec', 'hifigan', 'recon', 'ground_truth']

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 定义 resample 的函数
def resample_audio(input_file, output_file, target_rate=48000):
    command = [
        'ffmpeg',
        '-i', input_file,   # 输入文件
        '-ar', str(target_rate),  # 目标采样率
        output_file          # 输出文件
    ]
    subprocess.run(command, check=True)

# 为所有子文件夹生成相同的 100 个文件集合
all_files = os.listdir(os.path.join(base_path, 'encodec'))  # 假设所有子文件夹的文件集合一样
sampled_files = random.sample(all_files, 100)  # 随机抽取 100 个文件

# 处理每个子文件夹
for subfolder in subfolders:
    # 为子文件夹创建输出路径
    output_subfolder = os.path.join(output_folder, subfolder)
    os.makedirs(output_subfolder, exist_ok=True)

    # 对于其他文件夹，使用统一的 sampled_files
    files_to_process = sampled_files

    # 复制并重新采样文件到新文件夹
    for file_name in files_to_process:
        if subfolder == 'ground_truth':
            src_path = os.path.join(ground_truth_path, file_name)
        else:
            src_path = os.path.join(base_path, subfolder, file_name)

        dst_path = os.path.join(output_subfolder, file_name)
        
        # 使用 ffmpeg 重新采样并保存到目标路径
        resample_audio(src_path, dst_path)

print("文件已成功复制并重新采样到新文件夹结构中")
