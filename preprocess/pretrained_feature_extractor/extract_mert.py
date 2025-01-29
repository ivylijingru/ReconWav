"""A script to extract features from MERT model."""

import os
import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoModel
from data.data_utils import process_mert_format


mert_model = AutoModel.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
mert_model.eval()


def extract_mert_features(audio_path):
    """Extract MERT features from an audio file.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        dict: Dictionary containing the extracted features.
    """
    inputs = process_mert_format(audio_path, is_inference=True)
    with torch.no_grad():
        outputs = mert_model(**inputs, output_hidden_states=True)
    model_output = outputs.hidden_states[-1]
    model_output = model_output.transpose(-1, -2)
    return model_output


def extract_multiple_mert_features(input_dir, output_dir):
    """Extract MERT features from multiple audio files and save them to an output folder.

    Args:
        input_dir (str): Path to the folder containing audio files.
        output_dir (str): Path to the folder where features will be saved.

    Returns:
        None
    """
    input_file_list = [f for f in os.listdir(input_dir) if f.endswith((".wav", ".mp3"))]
    input_file_list = input_file_list[:2]  # HACK!!!

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for input_file in tqdm(input_file_list):
        input_path = os.path.join(input_dir, input_file)
        model_output = extract_mert_features(input_path)

        output_file = input_file.replace(".wav", ".npy")
        output_path = os.path.join(output_dir, output_file)
        np.save(output_path, model_output.cpu().numpy())


if __name__ == "__main__":
    import fire

    fire.Fire()
