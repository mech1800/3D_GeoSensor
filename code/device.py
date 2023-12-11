import torch

# 特定のCUDAデバイスのプロパティを取得
device_1 = torch.device("cuda:0")  # 0番目のGPUを選択
properties = torch.cuda.get_device_properties(device_1)

# プロパティ情報を表示
print(f"GPU名: {properties.name}")
print(f"メモリ容量: {properties.total_memory / 1024**2:.2f} MB")
print(f"計算能力: {properties.major}.{properties.minor}")
print("")

device_2 = torch.device("cuda:1")  # 0番目のGPUを選択
properties = torch.cuda.get_device_properties(device_2)

# プロパティ情報を表示
print(f"GPU名: {properties.name}")
print(f"メモリ容量: {properties.total_memory / 1024**2:.2f} MB")
print(f"計算能力: {properties.major}.{properties.minor}")