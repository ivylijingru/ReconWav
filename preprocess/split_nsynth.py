import os
import random
import json
random.seed(2022)


def write_input_json(root_dir, audio_dir):
    input_json = os.path.join(root_dir, "nsynth.json")
    input_json_f = open(input_json, "w")
    
    data_list = []
    for file_name in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, file_name)
        data = dict(wav_path=file_path)
        data_list.append(data)

    for data in data_list:
        json.dump(data, input_json_f)
        input_json_f.write('\n')
        input_json_f.flush()


def split_synth_train(root_dir, valid=0.15, test=0.15):
    """
    Split the openmic training data to separate training and valid

    Parameters
    ----------
    root_dir : str
        directory with magna.json
    valid : float
        percent of validation data
    test : float
        percent of test data
    """
    input_json = os.path.join(root_dir, 'nsynth.json')
    with open(input_json, 'r') as f:
        data = f.readlines()

    train_json = open(os.path.join(root_dir, 'nsynth_train.json'), 'w')
    valid_json = open(os.path.join(root_dir, 'nsynth_valid.json'), 'w')
    test_json = open(os.path.join(root_dir, 'nsynth_test.json'), 'w')
    for line in data:
        t = random.uniform(0., 1.)
        if t > valid + test:
            train_json.write(line)
        elif t < test:
            test_json.write(line)
        else:
            valid_json.write(line)
    
    train_json.close()
    valid_json.close()
    test_json.close()
    
    return


if __name__ == '__main__':
    root_dir = "../nsynth"
    write_input_json(root_dir,"../../nsynth-train/audio")
    split_synth_train(root_dir)