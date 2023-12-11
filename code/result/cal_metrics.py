import numpy as np
import torch
import os

import sys
sys.path.append('..')
from model import Encoder_Decoder_model

def cal_mse(outputs, labels):
    total_mse = 0
    total_count = 0
    for i in range(len(outputs)):
        mask = (outputs[i] > 0) & (labels[i] > 0)
        total_mse += np.sum((outputs[i][mask] - labels[i][mask]) ** 2)
        total_count += np.sum(mask)
    return total_mse / total_count if total_count > 0 else 0

def cal_mae(outputs, labels):
    total_mae = 0
    total_count = 0
    for i in range(len(outputs)):
        mask = (outputs[i] > 0) & (labels[i] > 0)
        total_mae += np.sum(np.abs(outputs[i][mask] - labels[i][mask]))
        total_count += np.sum(mask)
    return total_mae / total_count if total_count > 0 else 0

def cal_mre(outputs, labels):
    total_mre = 0
    total_count = 0
    for i in range(len(outputs)):
        mask = (outputs[i] > 0) & (labels[i] > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            mre = (np.abs(outputs[i][mask] - labels[i][mask])) / (10 + np.fmax(outputs[i][mask], labels[i][mask])) * 100
            mre[np.isnan(mre)] = 0  # Handle division by zero or NaNs
        total_mre += np.sum(mre)
        total_count += np.sum(mask)
    return total_mre / total_count if total_count > 0 else 0

def return_mse_mae_mre(count,train_or_test):
    # outputを1次元配列にしてリストに入れる
    outputs = []
    for i in range(count):
        # データをロードする
        contact_image = np.load('../../data/dataset/'+train_or_test+'/contact_images/' + str(i) + '.npy')
        depth_image = np.load('../../data/dataset/'+train_or_test+'/depth_images/' + str(i) + '.npy')
        geometry_image = np.load('../../data/dataset/'+train_or_test+'/geometry_images/' + str(i) + '.npy')
        initial_depth_image = np.load('../../data/dataset/'+train_or_test+'/initial_depth_images/' + str(i) + '.npy')
        initial_geometry_image = np.load('../../data/dataset/'+train_or_test+'/initial_geometry_images/' + str(i) + '.npy')

        # データ加工(depth_imageの非ゼロのピクセルを正規化する)
        unique_elements = np.unique(depth_image)
        depth_image[depth_image != 0] -= unique_elements[1]
        depth_image[depth_image != 0] /= (unique_elements[-1] - unique_elements[1])
        # データ加工(initial_depth_imageの非ゼロのピクセルを正規化する)
        unique_elements = np.unique(initial_depth_image)
        initial_depth_image[initial_depth_image != 0] -= unique_elements[1]
        initial_depth_image[initial_depth_image != 0] /= (unique_elements[-1] - unique_elements[1])

        data = np.stack((initial_geometry_image, geometry_image, initial_depth_image, depth_image, contact_image), axis=0)
        data = np.reshape(data, [1, data.shape[0], data.shape[1], data.shape[2]]).astype(np.float32)
        data = torch.from_numpy(data).to(device)

        output_i, _ = model(data)
        output_i = output_i.detach().cpu().numpy().ravel()

        # ハイパス&ローパスフィルタ
        output_i[output_i<150] = 0
        output_i[output_i>4000] = 4000

        outputs.append(output_i)
        print(i)

    # labelを1次元配列にしてリストに入れる
    labels = []
    for i in range(count):
        # ラベルをロードする
        force_image = np.load('../../data/dataset/'+train_or_test+'/force_images/' + str(i) + '.npy')
        label_i = force_image.ravel()
        label_i *= 1000   # データ加工(N→mN)
        labels.append(label_i)
        print(i)

    # mse,mae,mreの計算
    mse = cal_mse(outputs,labels)
    mae = cal_mae(outputs,labels)
    mre = cal_mre(outputs,labels)

    return mse,mae,mre


# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 学習済みモデルをロードする
model = Encoder_Decoder_model(inputDim=4, outputDim=1)
model = model.to(device)

# マルチGPUをONにする
'''
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    print('マルチGPUの使用をONにしました')
'''
model.load_state_dict(torch.load('./model_weight.pth'))

# forループを回すためのデータ数を取得する
file_list = os.listdir('../../data/dataset/train/contact_images/')
tr_file_count = len(file_list)
file_list = os.listdir('../../data/dataset/test/contact_images/')
va_file_count = len(file_list)

# 学習データに対する指標
tr_metrics = return_mse_mae_mre(tr_file_count,'train')
# テストデータに対する指標
va_metrics = return_mse_mae_mre(va_file_count,'test')

# textファイルに書き出し
f = open('metrics.txt','w')

f.write('training data\n')
f.write(str(tr_metrics[0])+'\n')
f.write(str(tr_metrics[1])+'\n')
f.write(str(tr_metrics[2])+'\n')

f.write('\n')

f.write('testing data\n')
f.write(str(va_metrics[0])+'\n')
f.write(str(va_metrics[1])+'\n')
f.write(str(va_metrics[2])+'\n')

f.close()