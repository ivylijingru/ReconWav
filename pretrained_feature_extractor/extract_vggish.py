import argparse
import os
from tqdm import tqdm
import soundfile as sf

import numpy as np
import torch

model = torch.hub.load(
    "harritaylor/torchvggish", "vggish", device="cuda", postprocess=True
)
model.eval()


def extract_vggish_feature(input_path, output_path, input_sr=44100):
    """
    Extract vggish file for one wavefile and save as an .npy file

    Parameters
    ----------
    input_path : str
        path to the audio file to extract the feature
    output_path : str
        path to save the output feature
    input_str : int
        sampling rate of the input file
    """
    output = model(input_path, fs=input_sr)
    # transpose output; to (feature_dim, time_dim)
    output = torch.transpose(output, 0, 1)
    output = output.cpu().detach().numpy().astype(np.int32)
    np.save(output_path, output)


def extract_multiple_vggish_feature(input_dir, output_dir, input_sr=16000):
    """
    Extract vggish file for multiple wav files and save as .npy files

    Parameters
    ----------
    input_dir : str
        directory of the input wav files
    output_dir : str
        directory of the output feature files
    input_str : int
        sampling rate of the input file
    """
    # input_file_list = os.listdir(input_dir)
    # Make sure that they are all waveforms
    input_file_list = [f for f in os.listdir(input_dir) if f.endswith((".wav", ".mp3"))]
    # input_file_list = input_file_list[:2]  # HACK!!!

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for input_file in tqdm(input_file_list):
        input_path = os.path.join(input_dir, input_file)

        # Make sure that the sample rate aligns
        audio, sampling_rate = sf.read(input_path)
        # assert sampling_rate == input_sr

        output_file = input_file.replace(".wav", ".npy")
        output_path = os.path.join(output_dir, output_file)

        extract_vggish_feature(
            input_path, output_path, sampling_rate
        )  # HACK: directly send in sampling_rate


if __name__ == "__main__":
    import fire

    fire.Fire()
