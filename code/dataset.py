import torch
from torch.utils.data import Dataset
import numpy as np

# dataとlabelをtensorのイテレータに変換するクラス
class MyDataset(Dataset):
    def __init__(self, dir_path, file, transform=None):
        super(MyDataset, self).__init__()

        self.dir_path = dir_path
        self.file = file
        self.transform = transform

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        # データを読み込む
        contact_image = np.load(self.dir_path[0]+self.file[idx])
        depth_image = np.load(self.dir_path[1]+self.file[idx])
        force_image = np.load(self.dir_path[2]+self.file[idx]) * 1000   # mNにする
        geometry_image = np.load(self.dir_path[3]+self.file[idx])
        initial_depth_image = np.load(self.dir_path[4]+self.file[idx])
        initial_geometry_image = np.load(self.dir_path[5]+self.file[idx])

        # depth_imageの非ゼロのピクセルを正規化する
        unique_elements = np.unique(depth_image)
        depth_image[depth_image != 0] -= unique_elements[1]
        depth_image[depth_image != 0] /= (unique_elements[-1]-unique_elements[1])

        # initial_depth_imageの非ゼロのピクセルを正規化する
        unique_elements = np.unique(initial_depth_image)
        initial_depth_image[initial_depth_image != 0] -= unique_elements[1]
        initial_depth_image[initial_depth_image != 0] /= (unique_elements[-1]-unique_elements[1])

        data = np.stack((initial_geometry_image,geometry_image,initial_depth_image,depth_image,contact_image),axis=0)
        label = np.reshape(force_image, [-1, force_image.shape[0], force_image.shape[1]])

        return data, label