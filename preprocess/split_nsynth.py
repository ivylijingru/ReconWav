import os
import random
import json

random.seed(2022)


def write_input_json(root_dir, audio_dir, mel_dir, name):
    input_json = os.path.join(root_dir, name)
    input_json_f = open(input_json, "w")

    data_list = []
    for file_name in os.listdir(audio_dir):
        wav_path = os.path.join(audio_dir, file_name)
        mel_path = os.path.join(mel_dir, file_name.split(".")[0] + "_mel.npy")
        assert os.path.exists(mel_path)
        data = dict(wav_path=wav_path, mel_path=mel_path)
        data_list.append(data)

    for data in data_list:
        json.dump(data, input_json_f)
        input_json_f.write("\n")
        input_json_f.flush()


if __name__ == "__main__":
    root_dir = "../nsynth"
    train_audio_dir = "../../nsynth-train/audio"
    valid_audio_dir = "../../nsynth-valid/audio"
    train_mel_dir = "mel_feature_train"
    valid_mel_dir = "mel_feature_valid"
    write_input_json(root_dir, train_audio_dir, train_mel_dir, "nsynth_train.json")
    write_input_json(root_dir, valid_audio_dir, valid_mel_dir, "nsynth_valid.json")
