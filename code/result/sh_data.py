import torch
import numpy as np
import torch.nn as nn
import open3d as o3d
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append('..')
from model import Encoder_Decoder_model

# データセットの何枚目を表示するか(number.npy)
number = 1500   # 表示したい画像のファイル名を入力

# rgb_image,depth_image,force_imageからカラー点群を作成する(事前にkinectの(fx,fy,cx,cy)が必要)
def make_color_pcd(rgb_image,depth_image,force_image):
    # depthイメージから点群を作成して座標(points)を抽出する
    depth = o3d.geometry.Image(depth_image)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(640,576,fx=504.2392272949219,fy=504.42327880859375,cx=321.95184326171875,cy=332.0074462890625)   # (int width, int height, double fx, double fy, double cx, double cy) kinectに固有の値を見つける必要がある
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsic)

    points = np.asarray(pcd.points)

    # depth_imageが0の要素はpcdになっていないので対応するrgb_imageの要素を消して(pcd.pointの要素数,3)のcolorsを抽出する
    depth_1d = depth_image.reshape(576*640)
    nonzero_depth_indices = np.nonzero(depth_1d)[0]

    rgb_image = rgb_image.astype(np.float64) / 255.0
    rgb_1d = rgb_image.reshape(576*640,3)
    # 背景のcolorを薄くしてforceのcolorを見やすくする
    # rgb_1d[:] = (224/255,224/255,224/255)   # 方法1:背景をグレー(224,224,224)にする
    # CV_RGB2RGBA CV"_RGBA2RGB   # 方法2:透明度1としてrgbaに変換した後に透明度を0.5にして最後にrgbに戻す
    colors = rgb_1d[nonzero_depth_indices]


    # force_imageの値が入っているピクセルをcolorsに反映する(矢印オブジェクトに隠れるので無くてもいい)
    force_1d = force_image.reshape(576*640)
    force = force_1d[nonzero_depth_indices]   # colorsと同じ形状にする

    nonzero_force_indices = np.nonzero(force)[0]   # 0でないforce値のindexを取得する

    '''
    cmap = plt.get_cmap("jet")
    max_force = np.max(force[nonzero_force_indices])   # force値を正規化するために最大値を取得する

    for index in nonzero_force_indices:
        rgba_value = cmap(force[index]/max_force)
        rgb_value = np.array(rgba_value[:3])
        colors[index] = rgb_value
    '''

    # pcdを新たに作成
    color_pcd = o3d.geometry.PointCloud()
    color_pcd.points = o3d.utility.Vector3dVector(points)
    color_pcd.colors = o3d.utility.Vector3dVector(colors)
    color_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return color_pcd, nonzero_force_indices, force


def visualization(pcd_output_final, nonzero_force_indices, nonzero_force, max_force):
    # 点群をコピーしてそれを使っても良い
    # new_pcd_output_final = pcd_output_final.clone()

    # 法線ベクトルを計算
    pcd_output_final.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=15))

    # 法線ベクトルの始点と終点を計算
    all_points = np.asarray(pcd_output_final.points)
    all_normals = np.asarray(pcd_output_final.normals)

    start_points = all_points[nonzero_force_indices]
    end_points = start_points + all_normals[nonzero_force_indices] * 0.1  # 0.1はただの計数

    '''
    # 法線ベクトルをラインセグメントとして作成する方法
    lines = []
    for i in range(len(nonzero_force_indices)):
        lines.append([i,len(nonzero_force_indices)+i])  # 各点から自身へのラインセグメント

    # 法線ベクトルの色を決める
    colors = np.array([[1,0,0] for _ in range(len(nonzero_force_indices))])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack((start_points, end_points)))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    '''

    # 法線ベクトルを矢印のジオメトリで作成する方法
    arrow_geometries = []
    for i in range(len(nonzero_force_indices)):
        arrow_start = start_points[i]
        arrow_end = end_points[i]
        arrow_direction = arrow_end - arrow_start
        arrow_length = np.linalg.norm(arrow_direction)

        # 矢印の3Dメッシュを作成
        relative_force_magnitude = nonzero_force[nonzero_force_indices[i]] / max_force   # 力の大きさを最大値で割った値(0~1)

        cylinder_height = arrow_length*relative_force_magnitude*0.8
        cone_height = arrow_length*0.1
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.002,
            cone_radius=0.004,
            cylinder_height=cylinder_height,
            cone_height=cone_height
        )

        # 矢印の始点を指定の座標に移動させる
        arrow.translate(arrow_start-(0,0,0))

        # 矢印の向きを逆にする
        center = np.array((0,0,(cylinder_height+cone_height)/2))+arrow_start   # 1step前で行った移動を考慮した矢印の中点
        arrow.rotate(R=o3d.geometry.get_rotation_matrix_from_xyz([-np.pi,0,0]), center=center)

        # 矢印を適切な方向に回転
        arrow.rotate(
            R=o3d.geometry.get_rotation_matrix_from_xyz(
                [-np.arcsin(arrow_direction[1] / arrow_length),
                 np.arctan2(arrow_direction[0], arrow_direction[2]),
                 0]
            ),
            center=arrow_start
        )

        '''
        # 矢印の末端が接触位置になるような描画する場合
        direction_vector = arrow_start-arrow_end
        normalized_direction = direction_vector / np.linalg.norm(direction_vector)
        length = cylinder_height+cone_height

        movement_vector = normalized_direction * length

        arrow.translate(movement_vector)
        '''

        # 力の大きさに応じた色を矢印オブジェクトに割り当てる
        cmap = plt.get_cmap("jet")
        rgba_value = cmap(relative_force_magnitude)
        rgb_value = np.array(rgba_value[:3])
        arrow.paint_uniform_color(rgb_value)

        arrow_geometries.append(arrow)

    # 可視化ウィンドウを作成
    # o3d.visualization.draw_geometries([pcd_output_final, line_set] + arrow_geometries)
    o3d.visualization.draw_geometries([pcd_output_final] + arrow_geometries)


# ----------モデルの出力を一例表示する----------

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

# データ・ラベル・表示用画像をロードする
contact_image = np.load('../../data/dataset/test/contact_images/'+str(number)+'.npy')
depth_image = np.load('../../data/dataset/test/depth_images/'+str(number)+'.npy')
geometry_image = np.load('../../data/dataset/test/geometry_images/'+str(number)+'.npy')
initial_depth_image = np.load('../../data/dataset/test/initial_depth_images/'+str(number)+'.npy')
initial_geometry_image = np.load('../../data/dataset/test/initial_geometry_images/'+str(number)+'.npy')
force_image = np.load('../../data/dataset/test/force_images/'+str(number)+'.npy')

original_depth_image = np.load('../../data/dataset/test/original_depth_images/'+str(number)+'.npy')
original_rgb_image = np.load('../../data/dataset/test/original_rgb_images/'+str(number)+'.npy')

# データ加工(N→mN)
force_image *= 1000
# データ加工(depth_imageの非ゼロのピクセルを正規化する)
unique_elements = np.unique(depth_image)
depth_image[depth_image != 0] -= unique_elements[1]
depth_image[depth_image != 0] /= (unique_elements[-1]-unique_elements[1])
# データ加工(initial_depth_imageの非ゼロのピクセルを正規化する)
unique_elements = np.unique(initial_depth_image)
initial_depth_image[initial_depth_image != 0] -= unique_elements[1]
initial_depth_image[initial_depth_image != 0] /= (unique_elements[-1]-unique_elements[1])
# データ加工(bgr→rgb)
original_rgb_image = original_rgb_image[:, :, ::-1]

# 学習済みモデルにデータを入力する
data = np.stack((initial_geometry_image,geometry_image,initial_depth_image,depth_image,contact_image),axis=0)
data = np.reshape(data, [1, data.shape[0], data.shape[1],data.shape[2]]).astype(np.float32)
data = torch.from_numpy(data).to(device)

output_final,output_prev = model(data)
output_final = output_final.detach().cpu().numpy()[0,0]
output_prev = output_prev.detach().cpu().numpy()[0,0]

# ハイパス&ローパスフィルタ
output_final[output_final<150] = 0
output_final[output_final>4000] = 4000

output_prev[output_prev<150] = 0
output_prev[output_prev>4000] = 4000

# 色を正規化するためにモデルの最終層の出力とラベルの最大値を取得する
max_force = np.max(np.maximum(output_final, force_image))

# モデルの最終層を表示・保存する
pcd_output_final, nonzero_force_indices, nonzero_force = make_color_pcd(original_rgb_image,original_depth_image,output_final)
visualization(pcd_output_final,nonzero_force_indices,nonzero_force,max_force)

# モデルの最終一層前の出力を表示・保存する
'''
pcd_output_prev = make_color_pcd(original_rgb_image,original_depth_image,output_prev)
visualization(pcd_output_prev,nonzero_force_indices,nonzero_force,max_force)
'''

# 正解ラベルを表示・保存する
pcd_label, nonzero_force_indices, nonzero_force = make_color_pcd(original_rgb_image,original_depth_image,force_image)
visualization(pcd_label,nonzero_force_indices,nonzero_force,max_force)


# ----------データセットの一例を表示する----------
'''
# FigureオブジェクトとAxesオブジェクトを作成
fig = plt.figure()

fig = plt.figure(figsize=(6,1))
fig.subplots_adjust(hspace=0.1, wspace=0.5)

ax1 = fig.add_subplot(1, 6, 1)
ax2 = fig.add_subplot(1, 6, 2)
ax3 = fig.add_subplot(1, 6, 3)
ax4 = fig.add_subplot(1, 6, 4)
ax5 = fig.add_subplot(1, 6, 5)
ax6 = fig.add_subplot(1, 6, 6)

# 画像をAxesオブジェクトに表示
ax1.imshow(initial_geometry_image)
ax2.imshow(geometry_image)
ax3.imshow(initial_depth_image)
ax4.imshow(depth_image)
ax5.imshow(contact_image)
ax6.imshow(force_image)

# 軸や目盛りを非表示にする
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
ax5.axis('off')
ax6.axis('off')

# タイトルを設定する
ax1.set_title("pre_geometry", loc='center', fontsize=10)
ax2.set_title("geometry", loc='center', fontsize=10)
ax3.set_title("pre_depth", loc='center', fontsize=10)
ax4.set_title("depth", loc='center', fontsize=10)
ax5.set_title("contact", loc='center', fontsize=10)
ax6.set_title("force", loc='center', fontsize=10)

plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)

# グラフを保存
plt.savefig('data.png')

# グラフを表示
plt.show()
'''