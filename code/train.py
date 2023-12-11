import torch
import torch.nn as nn
from torchvision import transforms
import os
from dataset import MyDataset
from torch.utils.data import DataLoader
from model import Encoder_Decoder_model
import torch.optim as optim
import matplotlib.pyplot as plt

# パラメータの設定
batch_size = 8
lr = 0.001
weight_decay=0.01
epochs = 100

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# メモリ節約のためにファイル名のリストをデータのように扱う
dir_path_tr = ['../data/dataset/train/contact_images/',
               '../data/dataset/train/depth_images/',
               '../data/dataset/train/force_images/',
               '../data/dataset/train/geometry_images/',
               '../data/dataset/train/initial_depth_images/',
               '../data/dataset/train/initial_geometry_images/']
file_tr = os.listdir(dir_path_tr[0])   # ファイル名はディレクトリ間で共有されているので代表してcontact_imagesからファイル名を取得する

train_dataset = MyDataset(dir_path=dir_path_tr, file=file_tr, transform=transforms.Compose([torch.tensor]))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


dir_path_va = ['../data/dataset/test/contact_images/',
               '../data/dataset/test/depth_images/',
               '../data/dataset/test/force_images/',
               '../data/dataset/test/geometry_images/',
               '../data/dataset/test/initial_depth_images/',
               '../data/dataset/test/initial_geometry_images/']
file_va = os.listdir(dir_path_va[0])   # ファイル名はディレクトリ間で共有されているので代表してcontact_imagesからファイル名を取得する

valid_dataset = MyDataset(dir_path=dir_path_va, file=file_va, transform=transforms.Compose([torch.tensor]))
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# 訓練用関数
def train(model, device, criterion, optimizer, trainloader):
    model.train()
    running_loss = 0

    for data, label in trainloader:
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output_final,output_prev = model(data)
        loss = criterion(output_final, label)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            running_loss += loss.item()

    return running_loss/len(trainloader)


# 評価用関数
def valid(model, device, criterion, validloader):
    model.eval()
    running_loss = 0

    for data, label in validloader:
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            output_final,output_prev = model(data)
            loss = criterion(output_final, label)
            running_loss += loss.item()

    return running_loss/len(validloader)


# モデル，評価関数，最適化関数を呼び出す
model = Encoder_Decoder_model(inputDim=4, outputDim=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)

# マルチGPUをONにする
'''
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    print('マルチGPUの使用をONにしました')
'''

# 指定したエポック数だけ学習指せる
tr_loss = []
va_loss = []

for epoch in range(1, 1+epochs):
    loss = train(model, device, criterion, optimizer, train_loader)
    tr_loss.append(loss)

    loss = valid(model, device, criterion, valid_loader)
    va_loss.append(loss)

    print(str(epoch)+'epoch通過')

else:
    torch.save(model.state_dict(), 'result/model_weight.pth')


# lossの推移をグラフにする
x = [i for i in range(epochs)]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, tr_loss, label='tr_loss')
ax.plot(x, va_loss, label='va_loss')
ax.set_xlabel('epoch')
ax.set_ylabel('MSE_loss')
ax.legend(loc='upper right')
fig.savefig('result/loss.png')
plt.show()