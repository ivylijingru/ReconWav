import os
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from data.data_utils import process_mert_format
from models import ReconstructModelMERTcb0

def load_model():
    device = 'cuda'
    checkpoint_path = "work_dir_mert/weight_new_data_cb0_net/epoch=19-val_loss-total=2.140.ckpt"
    recon_model = ReconstructModelMERTcb0.load_from_checkpoint(checkpoint_path).to(device)
    return recon_model

def calculate_accuracy(predictions, labels):
    """Calculates the accuracy of predictions.

    Args:
        predictions (torch.Tensor): Model predictions (after softmax).
        labels (torch.Tensor): True labels.

    Returns:
        float: Accuracy score.
    """
    predicted_classes = torch.argmax(predictions, dim=1)
    print(predicted_classes.shape, labels.shape)
    print(predicted_classes, labels)
    correct_predictions = (predicted_classes == labels).sum().item()
    total_predictions = labels.size(0)
    accuracy = correct_predictions / total_predictions
    return accuracy

def generate_audio(target_feature):
    from encodec import EncodecModel
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    target_feature = target_feature.unsqueeze(0)
    target_feature = target_feature.unsqueeze(0)
    print(target_feature.shape)
    out = model.decode([(target_feature, None)])
    ENCODEC_SR = 24000
    sf.write("help.wav", out.cpu().squeeze().detach().numpy(), ENCODEC_SR)

def generate_audio_predicted(predictions):
    from encodec import EncodecModel
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predicted_classes = torch.argmax(predictions, dim=1)
    target_feature = predicted_classes.squeeze()
    target_feature = target_feature.unsqueeze(0)
    target_feature = target_feature.unsqueeze(0)
    print(target_feature.shape)
    out = model.decode([(target_feature, None)])
    ENCODEC_SR = 24000
    sf.write("predicted.wav", out.cpu().squeeze().detach().numpy(), ENCODEC_SR)

def visualize_mert(X):
    # 可视化 2D 图片
    print(X.shape)
    plt.figure(figsize=(5, 10))  # 设置画布大小
    plt.imshow(X)  # 使用灰度颜色映射
    plt.title('2D Image Visualization')
    plt.axis('off')  # 关闭坐标轴

    # 保存为 PNG 文件
    plt.savefig('2d_image.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    cb0_test_dir = "../reconwav_data/encodec_cb0/test"
    nsynth_test_dir = "../nsynth/nsynth-test/audio/"

    model = load_model()
    for filename in os.listdir(nsynth_test_dir):
        file_path = os.path.join(nsynth_test_dir, filename)
        print(file_path)
        # break
        predicted_feature, mert_feature = model.inference_step(file_path)
        target_path = os.path.join(cb0_test_dir, filename.split(".")[0] + "_codebook0.npy")
        target_feature = torch.from_numpy(np.load(target_path)).long().cuda()
        print(calculate_accuracy(predicted_feature, target_feature))
        generate_audio(target_feature)
        generate_audio_predicted(predicted_feature)
        visualize_mert(mert_feature.squeeze().transpose(1, 0).cpu().numpy())
        break
