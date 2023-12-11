import numpy as np
import open3d as o3d
from scipy.ndimage import convolve

# depth画像を読み込む
depth_images = np.load('original_dataset/depth_images.npy')

# 点群の表示用
original_rgb_images = np.load('original_dataset/original_rgb_images.npy')

# グレースケール画像の形状を取得
height = 576
width = 640

# 補完を行う半径（周囲の範囲）を設定
radius = 1

# for i in range(1):
for i in range(depth_images.shape[0]):
    # i番目の画像を読み込み
    depth_image = depth_images[i]
    original_rgb_image = original_rgb_images[i]

    # 補間前の表示
    '''
    depth = o3d.geometry.Image(depth_image)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 576, fx=504.2392272949219, fy=504.42327880859375, cx=321.95184326171875, cy=332.0074462890625)  # (int width, int height, double fx, double fy, double cx, double cy) kinectに固有の値を見つける必要がある
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsic)
    points = np.asarray(pcd.points)

    depth_1d = depth_image.reshape(576 * 640)
    nonzero_depth_indices = np.nonzero(depth_1d)[0]
    rgb_image = original_rgb_image.astype(np.float64) / 255.0
    rgb_1d = rgb_image.reshape(576 * 640, 3)
    colors = rgb_1d[nonzero_depth_indices]

    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(points)
    pcd_new.colors = o3d.utility.Vector3dVector(colors)

    pcd_new.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd_new])
    '''


    for _ in range(3):

        # 0の値のピクセルを見つける
        zero_pixels = np.argwhere(depth_image == 0)

        # 各0のピクセルに対して補完を行う
        for zero_pixel in zero_pixels:
            row, col = zero_pixel

            # 周囲のピクセルを取得
            row_start = max(0, row - radius)
            row_end = min(height, row + radius + 1)
            col_start = max(0, col - radius)
            col_end = min(width, col + radius + 1)

            neighborhood = depth_image[row_start:row_end, col_start:col_end]

            # 0以外の値の平均を計算
            non_zero_values = neighborhood[neighborhood != 0]
            if non_zero_values.size > 5:
                average_value = np.mean(non_zero_values)
            else:
                # 周囲に0以外の値がない場合、0を代入
                average_value = 0

            # 補完
            depth_image[row, col] = average_value

    # 更新
    depth_images[i] = depth_image
    print(i)

    # 補間後の表示
    '''
    depth = o3d.geometry.Image(depth_image)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 576, fx=504.2392272949219, fy=504.42327880859375, cx=321.95184326171875, cy=332.0074462890625)  # (int width, int height, double fx, double fy, double cx, double cy) kinectに固有の値を見つける必要がある
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsic)
    points = np.asarray(pcd.points)

    depth_1d = depth_image.reshape(576 * 640)
    nonzero_depth_indices = np.nonzero(depth_1d)[0]
    rgb_image = original_rgb_image.astype(np.float64) / 255.0
    rgb_1d = rgb_image.reshape(576 * 640, 3)
    colors = rgb_1d[nonzero_depth_indices]

    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(points)
    pcd_new.colors = o3d.utility.Vector3dVector(colors)

    pcd_new.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd_new])
    '''

np.save('original_dataset/depth_images.npy', depth_images)