import os
import random
import json

random.seed(2022)


def write_input_json(root_dir, audio_dir, cb0_dir, name):
    input_json = os.path.join(root_dir, name)
    input_json_f = open(input_json, "w")

    data_list = []
    for file_name in os.listdir(audio_dir):
        wav_path = os.path.join(audio_dir, file_name)
        cb0_path = os.path.join(cb0_dir, file_name.split(".")[0] + "_codebook0.npy")
        assert os.path.exists(cb0_path)
        data = dict(wav_path=wav_path, cb0_path=cb0_path)
        data_list.append(data)

    for data in data_list:
        json.dump(data, input_json_f)
        input_json_f.write("\n")
        input_json_f.flush()


if __name__ == "__main__":
    root_dir = "../nsynth"
    train_audio_dir = "../../nsynth/nsynth-train/audio"
    valid_audio_dir = "../../nsynth-valid/audio"

    train_cb0_dir = "../../reconwav_data/encodec_cb0/train"
    valid_cb0_dir = "../../reconwav_data/encodec_cb0/valid"
    write_input_json(root_dir, train_audio_dir, train_cb0_dir, "nsynth_train_cb0.json")
    write_input_json(root_dir, valid_audio_dir, valid_cb0_dir, "nsynth_valid_cb0.json")
