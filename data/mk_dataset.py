import os
import pickle
import numpy as np
import cv2
import open3d as o3d
import pandas as pd

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

from utility import super_pixel, clustering,cluster_to_object, Extract_object

from openpose.src import util
from openpose.src.body import Body
from openpose.src.hand import Hand

def return_average_sensor_force(df_right,df_left,is_left,sensor_time,elem):
    if is_left:
        sensors_pd = df_left.loc[df_left['Time [ms]'] == sensor_time, elem]
    else:
        sensors_pd = df_right.loc[df_right['Time [ms]'] == sensor_time, elem]

    sensors = []
    for i in range(len(sensors_pd)):
        sensors.append(sensors_pd.iloc[i])
    sensor = sum(sensors)/len(sensors)

    return sensor


# データセット用に空のnumpy行列を用意しておく
initial_geometry_images = np.empty([0,576,640], dtype=np.float32)
initial_depth_images = np.empty([0,576,640], dtype=np.float32)
geometry_images = np.empty([0,576,640], dtype=np.float32)
depth_images = np.empty([0,576,640], dtype=np.float32)
contact_images = np.empty([0,576,640], dtype=np.float32)
force_images = np.empty([0,576,640], dtype=np.float32)

# 表示用に空のnumpy行列を用意しておく
original_rgb_images = np.empty([0,576,640,3], dtype=np.uint8)
original_depth_images = np.empty([0,576,640], dtype=np.int16)

# kinectが人物を認識していたシーンのファイル名を取得する
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
kinect_file_path = os.path.join(parent_dir, 'kinect_and_sensor/filename/kinect_times.dat')
jpg_file_path = os.path.join(parent_dir, 'kinect_and_sensor/filename/jpg_to_kinect.dat')
png_file_path = os.path.join(parent_dir, 'kinect_and_sensor/filename/png_to_kinect.dat')
sensor_file_path = os.path.join(parent_dir, 'kinect_and_sensor/filename/sensor_times_to_kinect.dat')

with open(kinect_file_path, 'rb') as file:
    kinect_times = pickle.load(file)
with open(jpg_file_path, 'rb') as file:
    jpg_to_kinect = pickle.load(file)
with open(png_file_path, 'rb') as file:
    png_to_kinect = pickle.load(file)
with open(sensor_file_path, 'rb') as file:
    sensor_times_to_kinect = pickle.load(file)

# sensorの値を取得する
sensor_file_path = os.path.join(parent_dir, 'kinect_and_sensor/balance_ball.csv')
df_right = pd.read_csv(sensor_file_path)
df_left = df_right.copy()   # 本来は左手のsensorデータをこの変数に入れる
df_left.iloc[:,1:] = 0

# openposeのdemoのimport位置を変えることで一つ上のディレクトリからopenposeを呼び出す
body_estimation = Body('openpose/model/body_pose_model.pth')
hand_estimation = Hand('openpose/model/hand_pose_model.pth')

# 各時刻ごとにデータセットを作成する
first_time = True

for i in range(len(kinect_times)):
    # データ量削減のために3回に1回のペースでデータセットに追加する
    if i % 3 != 0:
        continue

    # rgb画像を読み込む
    path = os.path.join(parent_dir, 'kinect_and_sensor/jpg/',jpg_to_kinect[i])
    bgr_image = cv2.imread(path)

    # depth画像を読み込む
    path = os.path.join(parent_dir, 'kinect_and_sensor/png/', png_to_kinect[i])
    original_depth_image = o3d.io.read_image(path)
    original_depth_image = np.array(original_depth_image, dtype=np.float32)

    '''
    # 物体の形状マスク(01)画像を抽出する(スーパーピクセル＋クラスタリング)
    geometry_image = Extract_object(bgr_image)
    geometry_image = np.array(geometry_image, dtype=np.float32)

    # depth画像も形状マスクと同じ領域にマスクする
    depth_image = original_depth_image*geometry_image
    '''

    # 指の関節点のピクセルを取得する(+周囲のピクセルもやってもいいかも)
    candidate, subset = body_estimation(bgr_image)
    hands_list = util.handDetect(candidate, subset, bgr_image)

    if hands_list == []:
        print('手が検知されなかったためスキップします')
        continue

    is_lefts = []
    all_hand_peaks = []

    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(bgr_image[y:y + w, x:x + w, :])
        peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
        peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)

        is_lefts.append(is_left)
        all_hand_peaks.append(peaks)

    # 物体の形状マスク(01)画像を抽出する(スーパーピクセル＋クラスタリング)
    bgr_image_copy = bgr_image.copy()
    geometry_image = Extract_object(bgr_image_copy)
    geometry_image = np.array(geometry_image, dtype=np.float32)

    # depth画像も形状マスクと同じ領域にマスクする
    depth_image = original_depth_image * geometry_image

    # このループのkinect_timeに対応するsensor_time
    sensor_time = sensor_times_to_kinect[i]

    # 指の関節点に力を与えたものをマスク画像にする
    force_image = np.zeros((576,640), dtype=np.float32)

    for j in range(len(all_hand_peaks)):
        is_left = is_lefts[j]
        force_image[tuple((all_hand_peaks[j][0,1],all_hand_peaks[j][0,0]))] = return_average_sensor_force(df_right,df_left,is_left,sensor_time,' Elem5')
        force_image[tuple((all_hand_peaks[j][1,1],all_hand_peaks[j][1,0]))] = 0
        force_image[tuple((all_hand_peaks[j][2,1],all_hand_peaks[j][2,0]))] = return_average_sensor_force(df_right,df_left,is_left,sensor_time,' Elem6')
        force_image[tuple((all_hand_peaks[j][3,1],all_hand_peaks[j][3,0]))] = 0
        force_image[tuple((all_hand_peaks[j][4,1],all_hand_peaks[j][4,0]))] = return_average_sensor_force(df_right,df_left,is_left,sensor_time,' Elem0')
        force_image[tuple((all_hand_peaks[j][5,1],all_hand_peaks[j][5,0]))] = 0
        force_image[tuple((all_hand_peaks[j][6,1],all_hand_peaks[j][6,0]))] = return_average_sensor_force(df_right,df_left,is_left,sensor_time,' Elem7')
        force_image[tuple((all_hand_peaks[j][7,1],all_hand_peaks[j][7,0]))] = 0
        force_image[tuple((all_hand_peaks[j][8,1],all_hand_peaks[j][8,0]))] = return_average_sensor_force(df_right,df_left,is_left,sensor_time,' Elem1')
        force_image[tuple((all_hand_peaks[j][9,1],all_hand_peaks[j][9,0]))] = 0
        force_image[tuple((all_hand_peaks[j][10,1],all_hand_peaks[j][10,0]))] = return_average_sensor_force(df_right,df_left,is_left,sensor_time,' Elem8')
        force_image[tuple((all_hand_peaks[j][11,1],all_hand_peaks[j][11,0]))] = 0
        force_image[tuple((all_hand_peaks[j][12,1],all_hand_peaks[j][12,0]))] = return_average_sensor_force(df_right,df_left,is_left,sensor_time,' Elem2')
        force_image[tuple((all_hand_peaks[j][13,1],all_hand_peaks[j][13,0]))] = 0
        force_image[tuple((all_hand_peaks[j][14,1],all_hand_peaks[j][14,0]))] = return_average_sensor_force(df_right,df_left,is_left,sensor_time,' Elem9')
        force_image[tuple((all_hand_peaks[j][15,1],all_hand_peaks[j][15,0]))] = 0
        force_image[tuple((all_hand_peaks[j][16,1],all_hand_peaks[j][16,0]))] = return_average_sensor_force(df_right,df_left,is_left,sensor_time,' Elem3')
        force_image[tuple((all_hand_peaks[j][17,1],all_hand_peaks[j][17,0]))] = 0
        force_image[tuple((all_hand_peaks[j][18,1],all_hand_peaks[j][18,0]))] = return_average_sensor_force(df_right,df_left,is_left,sensor_time,' Elem10')
        force_image[tuple((all_hand_peaks[j][19,1],all_hand_peaks[j][19,0]))] = 0
        force_image[tuple((all_hand_peaks[j][20,1],all_hand_peaks[j][20,0]))] = return_average_sensor_force(df_right,df_left,is_left,sensor_time,' Elem4')

    # 検知出来なかった関節点[0,0]がある場合にその関節点に割り当てられたセンサ値を0に戻す
    force_image[0,0] = 0

    # 0.2以下の要素は0にする
    force_image[force_image<0.2] = 0

    # 全ての要素が0ならcontinueする
    if np.all(force_image==0):
        print('センサ値が全て0なのでスキップします')
        continue

    # 異常値が検出された場合はcontinueする
    if np.max(force_image)>8:
        print(f'異常値が検出されましたのでスキップします 最大値は{np.max(force_image)}です')
        continue

    '''
    # 数値が入ってる周りの8マスにも同じ値を入れる
    original_force_image = np.copy(force_image)

    up = np.zeros((576,640), dtype=np.float32)
    up[:-1, :] = original_force_image[1:, :]
    down = np.zeros((576,640), dtype=np.float32)
    down[1:, :] = original_force_image[:-1, :]
    left = np.zeros((576,640), dtype=np.float32)
    left[:, :-1] = original_force_image[:, 1:]
    right = np.zeros((576,640), dtype=np.float32)
    right[:, 1:] = original_force_image[:, :-1]

    up_left = np.zeros((576,640), dtype=np.float32)
    up_left[:-1, :-1] = original_force_image[1:, 1:]
    up_right = np.zeros((576,640), dtype=np.float32)
    up_right[:-1, 1:] = original_force_image[1:, :-1]
    down_left = np.zeros((576,640), dtype=np.float32)
    down_left[1:, :-1] = original_force_image[:-1, 1:]
    down_right = np.zeros((576,640), dtype=np.float32)
    down_right[1:, 1:] = original_force_image[:-1, :-1]

    force_image += up +down + left + right + up_left + up_right + down_left + down_right
    '''

    # 01のcontact画像を作成する
    contact_image = np.zeros((576, 640), dtype=np.float32)

    for j in range(len(all_hand_peaks)):
        if is_lefts[j] == True:
            continue
        contact_image[tuple((all_hand_peaks[j][0,1],all_hand_peaks[j][0,0]))] = 1
        contact_image[tuple((all_hand_peaks[j][1,1],all_hand_peaks[j][1,0]))] = 0
        contact_image[tuple((all_hand_peaks[j][2,1],all_hand_peaks[j][2,0]))] = 1
        contact_image[tuple((all_hand_peaks[j][3,1],all_hand_peaks[j][3,0]))] = 0
        contact_image[tuple((all_hand_peaks[j][4,1],all_hand_peaks[j][4,0]))] = 1
        contact_image[tuple((all_hand_peaks[j][5,1],all_hand_peaks[j][5,0]))] = 0
        contact_image[tuple((all_hand_peaks[j][6,1],all_hand_peaks[j][6,0]))] = 1
        contact_image[tuple((all_hand_peaks[j][7,1],all_hand_peaks[j][7,0]))] = 0
        contact_image[tuple((all_hand_peaks[j][8,1],all_hand_peaks[j][8,0]))] = 1
        contact_image[tuple((all_hand_peaks[j][9,1],all_hand_peaks[j][9,0]))] = 0
        contact_image[tuple((all_hand_peaks[j][10,1],all_hand_peaks[j][10,0]))] = 1
        contact_image[tuple((all_hand_peaks[j][11,1],all_hand_peaks[j][11,0]))] = 0
        contact_image[tuple((all_hand_peaks[j][12,1],all_hand_peaks[j][12,0]))] = 1
        contact_image[tuple((all_hand_peaks[j][13,1],all_hand_peaks[j][13,0]))] = 0
        contact_image[tuple((all_hand_peaks[j][14,1],all_hand_peaks[j][14,0]))] = 1
        contact_image[tuple((all_hand_peaks[j][15,1],all_hand_peaks[j][15,0]))] = 0
        contact_image[tuple((all_hand_peaks[j][16,1],all_hand_peaks[j][16,0]))] = 1
        contact_image[tuple((all_hand_peaks[j][17,1],all_hand_peaks[j][17,0]))] = 0
        contact_image[tuple((all_hand_peaks[j][18,1],all_hand_peaks[j][18,0]))] = 1
        contact_image[tuple((all_hand_peaks[j][19,1],all_hand_peaks[j][19,0]))] = 0
        contact_image[tuple((all_hand_peaks[j][20,1],all_hand_peaks[j][20,0]))] = 1

    if first_time:
        initial_geometry_image = geometry_image
        initial_depth_image = depth_image
        first_time = False

    '''
    # 画像で確認
    bgr_image = bgr_image.astype(np.uint8)
    original_depth_image = original_depth_image.astype(np.uint8)

    initial_geometry_image = initial_geometry_image.astype(np.uint8)
    initial_geometry_image *= 100
    initial_depth_image = initial_depth_image.astype(np.uint8)
    geometry_image = geometry_image.astype(np.uint8)
    geometry_image *= 100
    depth_image = depth_image.astype(np.uint8)
    contact_image = contact_image.astype(np.uint8)
    contact_image *= 100
    force_image = force_image.astype(np.uint8)

    cv2.imshow('1', bgr_image)
    cv2.waitKey(0)
    cv2.imshow('2', original_depth_image)
    cv2.waitKey(0)

    cv2.imshow('3', initial_geometry_image)
    cv2.waitKey(0)
    cv2.imshow('4', initial_depth_image)
    cv2.waitKey(0)
    cv2.imshow('5', geometry_image)
    cv2.waitKey(0)
    cv2.imshow('6', depth_image)
    cv2.waitKey(0)
    cv2.imshow('7', contact_image)
    cv2.waitKey(0)
    cv2.imshow('8', force_image)
    cv2.waitKey(0)

    cv2.destroyWindow('1')
    cv2.destroyWindow('2')
    cv2.destroyWindow('3')
    cv2.destroyWindow('4')
    cv2.destroyWindow('5')
    cv2.destroyWindow('6')
    cv2.destroyWindow('7')
    cv2.destroyWindow('8')
    '''

    # データ上書き
    initial_geometry_images = np.vstack((initial_geometry_images, [initial_geometry_image]))
    initial_depth_images = np.vstack((initial_depth_images, [initial_depth_image]))
    geometry_images = np.vstack((geometry_images, [geometry_image]))
    depth_images = np.vstack((depth_images, [depth_image]))
    contact_images = np.vstack((contact_images, [contact_image]))
    force_images = np.vstack((force_images, [force_image]))

    original_rgb_images = np.vstack((original_rgb_images, [bgr_image]))  # rgbに直しても良いがその場合はopenpose用にbgrに戻さなければならない
    original_depth_image = original_depth_image.astype(np.int16)
    original_depth_images = np.vstack((original_depth_images, [original_depth_image]))

    print(i)

# データ書き出し
np.save('original_dataset/initial_geometry_images.npy', initial_geometry_images)
np.save('original_dataset/initial_depth_images.npy', initial_depth_images)
np.save('original_dataset/geometry_images.npy', geometry_images)
np.save('original_dataset/depth_images.npy', depth_images)
np.save('original_dataset/contact_images.npy', contact_images)
np.save('original_dataset/force_images.npy', force_images)

np.save('original_dataset/original_rgb_images.npy', original_rgb_images)
np.save('original_dataset/original_depth_images.npy', original_depth_images)