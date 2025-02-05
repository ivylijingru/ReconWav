import os
import sys
from tqdm import tqdm
import numpy as np

parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_folder)
from preprocess.pretrained_feature_extractor.extract_encodec import (
    extract_multiple_encodec_feature,
)
from preprocess.pretrained_feature_extractor.extract_vggish import (
    extract_multiple_vggish_feature,
)
from preprocess.pretrained_feature_extractor.extract_mert import (
    extract_multiple_mert_features,
)


def generate_output_dirs(input_dir_dict, features, base_output_dir="../features"):
    """
    Generate a nested dictionary of output directories for each input directory and feature.

    Args:
        input_dirs (list): List of input wave directories.
        features (list): List of feature names to extract.
        base_output_dir (str): Base directory to store feature outputs.

    Returns:
        dict: Nested dictionary mapping input directories to feature output directories.
    """
    output_dirs = {}
    for wav_type in input_dir_dict.keys():
        output_dirs[wav_type] = {
            feature: os.path.join(base_output_dir, wav_type, feature)
            for feature in features
        }
    return output_dirs


def extract_features(input_dir_dict, output_dir_dict, features, extractors):
    """
    Extract multiple features for each input wave directory and save to respective output directories.

    Args:
        input_dirs (list): List of input wave directories.
        output_dirs (list): List of output feature directories.
        features (list): List of feature names to extract.
        extractors (dict): Dictionary mapping feature names to their extraction functions.
    """
    for wav_type in input_dir_dict.keys():
        input_dir = input_dir_dict[wav_type]
        output_feature_dict = output_dir_dict[wav_type]

        if not os.path.exists(input_dir):
            print(f"Input directory {input_dir} does not exist. Skipping.")
            continue

        for feature in features:
            if feature not in extractors:
                print(f"No extractor defined for feature {feature}. Skipping.")
                continue

            extract_fn = extractors[feature]
            output_dir = output_feature_dict[feature]

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print(f"Extracting {feature} features from {input_dir} to {output_dir}...")

            extract_fn(input_dir, output_dir)  # Assume a default sampling rate of 16kHz


def ensure_common_files(dir1, dir2):
    """
    Ensure that two directories have common files.

    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.

    Returns:
        bool: True if common files exist, False otherwise.
    """
    valid_flag = True
    if not os.path.exists(dir1) or not os.path.exists(dir2):
        print(f"Feature directories for {feature} do not exist. Skipping.")
        valid_flag = False

    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))
    common_files = files1.intersection(files2)

    if not common_files:
        print(f"No common files found for feature {feature}. Skipping.")
        valid_flag = False

    return valid_flag, common_files


def calculate_mean_difference(common_files, dir1, dir2):
    """
    Calculate the mean difference between two sets of features.

    Args:
        common_files (list): List of common feature files to compare.
        dir1 (str): Path to the first set of feature files.
        dir2 (str): Path to the second set of feature files.

    Returns:
        float: Mean difference between the two sets of features.
    """
    diffs = []
    for file in tqdm(common_files):
        path1 = os.path.join(dir1, file)
        path2 = os.path.join(dir2, file)

        data1 = np.load(path1)
        data2 = np.load(path2)

        diff = np.linalg.norm(data1 - data2)
        diffs.append(diff)

        import matplotlib.pyplot as plt
        import seaborn as sns

        # 使用 seaborn 画 KDE
        sns.kdeplot(diffs, fill=True, color="blue", alpha=0.5)
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title("KDE Plot of Data Distribution")
        plt.savefig(
            f"histogram_{dir1}_{dir2}.png", dpi=300, bbox_inches="tight"
        )  # dpi=300 提高清晰度

    # analyze the feature difference by mean, variance, etc.
    dict_diff = {
        "mean": np.mean(diffs),
        "variance": np.var(diffs),
        "max": np.max(diffs),
        "min": np.min(diffs),
    }

    return dict_diff


def calculate_feature_differences(wav_type_list, output_dir_dict, features):
    """
    Calculate the differences between features of two input directories.

    Args:
        wav_type_list (list): List containing two input wave directories.
        input_dirs (list): List containing two input wave directories.
        features (list): List of feature names to calculate differences for.

    Returns:
        dict: Dictionary with average differences for each feature.
    """
    if len(wav_type_list) != 2:
        raise ValueError("Exactly two input directories are required for comparison.")

    differences = {}

    for feature in features:
        dir1 = output_dir_dict[wav_type_list[0]][feature]
        dir2 = output_dir_dict[wav_type_list[1]][feature]

        valid_flag, common_files = ensure_common_files(dir1, dir2)
        if not valid_flag:
            continue

        differences[feature] = calculate_mean_difference(common_files, dir1, dir2)

    return differences


if __name__ == "__main__":

    input_dir_dict = {
        "origin": "../../nsynth-valid/audio/",
        "encodec": "../samples-new-hifi/encodec",
        "hifigan": "../samples-new-hifi/hifigan",
        "mert_recon": "../samples-bigvgan-mert-128/recon",
    }

    features = ["Encodec", "VGGish", "MERT"]

    extractors = {
        "Encodec": extract_multiple_encodec_feature,
        "VGGish": extract_multiple_vggish_feature,
        "MERT": extract_multiple_mert_features,
    }

    output_dir_dict = generate_output_dirs(
        input_dir_dict, features, base_output_dir="../features"
    )
    extract_features(input_dir_dict, output_dir_dict, features, extractors)
    input_wav_type_list = list(input_dir_dict.keys())

    for wav_type in input_wav_type_list[1:]:
        wav_type_list = [input_wav_type_list[0], wav_type]
        # [reference index, current index]
        differences = calculate_feature_differences(
            wav_type_list, output_dir_dict, features
        )

        print(f"{wav_type}:")
        for feature, diff_dict in differences.items():
            for key, value in diff_dict.items():
                print(f"{key} for {feature}: {value}")
