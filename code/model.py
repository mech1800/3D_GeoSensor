import torch
import torch.nn as nn

# Encoder_DecoderクラスのためにBasicBlockクラスを作成する
class BasicBlock(nn.Module):
    expansion = 1  # 出力のチャンネル数を入力のチャンネル数の何倍に拡大するか

    def __init__(self, inputDim, outputDim, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inputDim, outputDim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputDim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inputDim, outputDim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outputDim)

        # 入力と出力のチャンネル数が異なる場合、x をダウンサンプリングする。
        if inputDim != outputDim * self.expansion:
            self.shortcut = nn.Sequential(nn.Conv2d(inputDim, outputDim*self.expansion, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(outputDim*self.expansion))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out


# force用のモデル
class Encoder_Decoder_model(nn.Module):
    def __init__(self, inputDim, outputDim, dropout_rate_encoder=0.5, dropout_rate_decoder=0.2):
        super(Encoder_Decoder_model,  self).__init__()

        # encoder
        self.conv1 = nn.Sequential(nn.Conv2d(inputDim, 32, kernel_size=9, stride=1, padding=4), nn.BatchNorm2d(32), nn.LeakyReLU(), nn.Dropout(dropout_rate_encoder))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(), nn.Dropout(dropout_rate_encoder))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(), nn.Dropout(dropout_rate_encoder))

        # resnet
        self.RN1 = BasicBlock(128, 128)
        self.RN2 = BasicBlock(128, 128)
        self.RN3 = BasicBlock(128, 128)
        self.RN4 = BasicBlock(128, 128)
        self.RN5 = BasicBlock(128, 128)

        # decoder
        self.conv4 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(dropout_rate_decoder))
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout(dropout_rate_decoder))
        self.conv6 = nn.Sequential(nn.Conv2d(32, outputDim, kernel_size=9, stride=1, padding=4), nn.ReLU())

    def forward(self, x):
        # encoder
        out = self.conv1(x[:,:4,:,:])
        out = self.conv2(out)
        out = self.conv3(out)

        # resnet
        out = self.RN1(out)
        out = self.RN2(out)
        out = self.RN3(out)
        out = self.RN4(out)
        out = self.RN5(out)

        # decoder
        out = self.conv4(out)
        out = self.conv5(out)
        out_prev = self.conv6(out)

        out_final = out_prev * x[:,4:5,:,:]

        return out_final,out_prev