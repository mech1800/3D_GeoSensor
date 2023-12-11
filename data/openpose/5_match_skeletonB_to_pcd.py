import cv2
from src.body import Body

import pandas as pd
import numpy as np
import open3d as o3d

import os
import pickle

# csv,jpg,pngのファイル名を取得する
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
csv_file_path = os.path.join(parent_dir, 'mk_dataset_1/filename/csv_times.dat')
jpg_file_path = os.path.join(parent_dir, 'mk_dataset_1/filename/jpg_camera2_to_csv.dat')
png_file_path = os.path.join(parent_dir, 'mk_dataset_1/filename/png_camera2_to_csv.dat')

with open(csv_file_path, 'rb') as file:
    csv_times = pickle.load(file)

with open(jpg_file_path, 'rb') as file:
    jpg_camera2_to_csv = pickle.load(file)

with open(png_file_path, 'rb') as file:
    png_camera2_to_csv = pickle.load(file)

# openposeのモデルを読み込む
body_estimation = Body('model/body_pose_model.pth')

# openposeの骨格座標を使う方法
for i in range(len(csv_times)):
    # jpg画像を読み込む
    jpg_path = os.path.join(parent_dir, 'mk_dataset_1/camera_2/20230926182724_127000000001/jpg_to_png/', jpg_camera2_to_csv[i])
    jpg_image = cv2.imread(jpg_path)  # B,G,R order
    # jpg_image = o3d.io.read_image(jpg_path)
    # jpg_image = np.array(jpg_image, dtype=np.uint8)

    # png画像を読み込む
    png_path = os.path.join(parent_dir, 'mk_dataset_1/camera_2/20230926182724_127000000001/png/', png_camera2_to_csv[i])
    png_image = o3d.io.read_image(png_path)
    png_image = np.array(png_image, dtype=np.uint16)

    # jpgに対してopenposeを行って手の骨格座標のピクセルを特定する
    candidate, subset = body_estimation(jpg_image)

    # right_wrist_index = np.count_nonzero(subset[0, :4] != -1)
    # right_wrist_xy = candidate[right_wrist_index, :2] if subset[0, 4] != -1 else np.array([0, 0], dtype=np.float64)

    left_wrist_index = np.count_nonzero(subset[0, :7] != -1)
    left_wrist_xy = candidate[left_wrist_index, :2] if subset[0, 7] != -1 else np.array([0, 0], dtype=np.float64)

    # そのピクセルに対応するpng(depth)のピクセル値が0なら次の画像に移行する
    if png_image[int(left_wrist_xy[1]), int(left_wrist_xy[0])] == 0:
        print('左手の深度情報なし')
        continue

    # 画像を1次元にreshapeして確定したピクセルのindexを取得する
    else:
        # right_wrist_xyのindexまでに0でない(深度情報がある)ピクセルが何個あるか調べる
        png_1d = png_image.reshape(576 * 640)
        index = int((left_wrist_xy[1]-1)*640+left_wrist_xy[0])   # 1ズレてる可能性大
        true_index = np.count_nonzero(png_1d[:index] != 0)

    # 画像を1次元にreshapeして確定したピクセルのindexを取得する

    # indexを用いて手の座標を取得する
    pcd2_path = os.path.join(parent_dir, 'mk_dataset_1/pcd_2/', str(i)+'.pcd')
    pcd2 = o3d.io.read_point_cloud(pcd2_path)
    # 手の座標になっているか確認
    pcd2.colors[true_index] = [255,0,0]
    o3d.visualization.draw_geometries([pcd2])

    left_wrist_xyz = pcd2.points[true_index]

    # バランスボール点群の中で最も手の座標に近い点を色付ける
    pcd_path = os.path.join(parent_dir, 'mk_dataset_1/pcd/', str(i)+'.pcd')
    pcd = o3d.io.read_point_cloud(pcd_path)
    target_points = []
    target_points.append(left_wrist_xyz)

    for target_point in target_points:
        # KDツリーを構築してターゲットの座標に最も近い点を取得する
        kd_tree = o3d.geometry.KDTreeFlann(pcd)
        k, idx, dist = kd_tree.search_knn_vector_3d(target_point, 3)   # 近傍の3点を取得する

        # 近傍点が遠すぎる場合は近傍点とみなさない
        if dist[0] > 99999:
            continue

        nearest_point_index = idx[0]

        # 色情報を変更
        new_color = [255, 0, 0]
        pcd.colors[nearest_point_index] = new_color
        o3d.visualization.draw_geometries([pcd])

'''
# kinectの骨格座標を使う方法(点群の座標系とkinectの座標系を合わせる必要がある)
csv_path = 'camera_2/20230828213915_127000000001_y/pos_20230828213915.csv'
df = pd.read_csv(csv_path)

# for i in range(len(csv_times)):
for i in range(1):
    # 点群を読み込む
    # pcd = o3d.io.read_point_cloud("pcd/"+str(i)+".pcd")
    pcd = o3d.io.read_point_cloud("pcd_2/"+str(i)+".pcd")

    # 右の手の平(hand_right),手の先(handtip_right),親指(thumb_right)の座標
    hand_right = np.array([df.iloc[i,47], df.iloc[i,48], df.iloc[i,49]])
    handtip_right = np.array([df.iloc[i,50],df.iloc[i,51],df.iloc[i,52]])
    thumb_right = np.array([df.iloc[i,53], df.iloc[i,54], df.iloc[i,55]])
    # 左の手の平(hand_left),手の先(handtip_left),親指(thumb_left)の座標
    hand_left = np.array([df.iloc[i,26], df.iloc[i,27], df.iloc[i,28]])
    handtip_left = np.array([df.iloc[i,29], df.iloc[i,30], df.iloc[i,31]])
    thumb_left = np.array([df.iloc[i,32], df.iloc[i,33], df.iloc[i,34]])

    target_points = [hand_right, handtip_right, thumb_right, hand_left, handtip_left, thumb_left]

    # 点群の中から手の骨格座標に最も近い数点に色を着ける
    for target_point in target_points:
        # KDツリーを構築してターゲットの座標に最も近い点を取得する
        kd_tree = o3d.geometry.KDTreeFlann(pcd)
        k, idx, dist = kd_tree.search_knn_vector_3d(target_point, 3)

        # 近傍点が遠すぎる場合は近傍点とみなさない
        if dist[0] > 99999:
            continue

        nearest_point_index = idx[0]

        # 色情報を変更
        new_color = [0,0,255]
        pcd.colors[nearest_point_index] = new_color

    o3d.visualization.draw_geometries([pcd])
'''