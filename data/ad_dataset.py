# 個別のnpyファイルとしてまとめる方法
import numpy as np
from scipy import ndimage
import math
import os

'''
# test1
contact_image = np.zeros((100, 100))
contact_image[49,49] = 1
modified_contact_image = ndimage.zoom(contact_image, 1.5, order=0)
nonzero_indices = np.nonzero(modified_contact_image)
nonzero_indices = tuple(zip(nonzero_indices[0], nonzero_indices[1]))
for nonzero_index in nonzero_indices:
    i, j = nonzero_index
    if modified_contact_image[i, j] == 0:
        continue
    modified_contact_image[i-1,j-1] = 0
    modified_contact_image[i-1,j] = 0
    modified_contact_image[i-1,j+1] = 0
    modified_contact_image[i,j-1] = 0
    modified_contact_image[i,j+1] = 0
    modified_contact_image[i+1,j-1] = 0
    modified_contact_image[i+1,j] = 0
    modified_contact_image[i+1,j+1] = 0

force_image = np.zeros((100, 100))
force_image[49,49] = 3.2
modified_force_image = ndimage.zoom(force_image, 1.5, order=0)
modified_force_image *= modified_contact_image

# test2
# 1000枚の100x100の画像が入ったnumpy行列を仮定
images = np.random.random((1000, 100, 100))
# 各画像ごとに0より大きい値の数を数える
counts = np.sum(images > 0, axis=(1, 2))
# 最も多い個数とそのインデックスを取得
max_count = np.max(counts)
max_index = np.argmax(counts)
'''

def determine_offset(scale):
    LR = np.random.uniform(-320*(scale-1), 320*(scale-1))
    # 絶対値が小さくなるように切り捨て
    if LR>0:
        LR = math.floor(LR)
    else:
        LR = math.ceil(LR)

    UD = np.random.uniform(-288*(scale-1), 288*(scale-1))
    # 絶対値が小さくなるように切り捨て
    if UD>0:
        UD = math.floor(UD)
    else:
        UD = math.ceil(UD)

    return (LR, UD)

def randam_expand_and_slide(
        initial_geometry_image,
        initial_depth_image,
        geometry_image,
        depth_image,
        contact_image,
        force_image,
        original_rgb_image,
        original_depth_image):

    # rgbを各チャンネルに分割して処理して最後に3チャンネルに結合する
    original_r_image = original_rgb_image[:,:,0]
    original_g_image = original_rgb_image[:,:,1]
    original_b_image = original_rgb_image[:,:,2]

    # 画像をランダムな倍数で拡大する
    random_scale = np.random.uniform(1, 1.2)

    modified_initial_geometry_image = ndimage.zoom(initial_geometry_image, random_scale, order=0)
    modified_initial_depth_image = ndimage.zoom(initial_depth_image, random_scale, order=0)
    modified_geometry_image = ndimage.zoom(geometry_image, random_scale, order=0)
    modified_depth_image = ndimage.zoom(depth_image, random_scale, order=0)
    modified_contact_image = ndimage.zoom(contact_image, random_scale, order=0)
    modified_force_image = ndimage.zoom(force_image, random_scale, order=0)

    modified_original_r_image = ndimage.zoom(original_r_image, random_scale, order=0)
    modified_original_g_image = ndimage.zoom(original_g_image, random_scale, order=0)
    modified_original_b_image = ndimage.zoom(original_b_image, random_scale, order=0)
    modified_original_depth_image = ndimage.zoom(original_depth_image, random_scale, order=0)

    # 画像をランダムなオフセットでスライドさせる
    offset = determine_offset(random_scale)

    # 拡大した画像から中心からランダムなオフセットの分だけずらして640×576ピクセルを抽出する
    start_x = (modified_initial_geometry_image.shape[1]-640)//2
    start_y = (modified_initial_geometry_image.shape[0]-576)//2

    modified_initial_geometry_image = modified_initial_geometry_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]
    modified_initial_depth_image = modified_initial_depth_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]
    modified_geometry_image = modified_geometry_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]
    modified_depth_image = modified_depth_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]
    modified_contact_image = modified_contact_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]
    modified_force_image = modified_force_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]

    modified_original_r_image = modified_original_r_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]
    modified_original_g_image = modified_original_g_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]
    modified_original_b_image = modified_original_b_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]
    modified_original_rgb_image = np.stack([modified_original_r_image, modified_original_g_image, modified_original_b_image], axis=2)
    modified_original_depth_image = modified_original_depth_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]

    # force_imageとcontact_imageは横に連続する1のピクセルを消去する
    nonzero_indices = np.nonzero(modified_contact_image)
    nonzero_indices = tuple(zip(nonzero_indices[0], nonzero_indices[1]))
    for nonzero_index in nonzero_indices:
        i, j = nonzero_index
        if modified_contact_image[i, j] == 0:
          continue
        modified_contact_image[i-1,j-1] = 0
        modified_contact_image[i-1,j] = 0
        modified_contact_image[i-1,j+1] = 0
        modified_contact_image[i,j-1] = 0
        modified_contact_image[i,j+1] = 0
        modified_contact_image[i+1,j-1] = 0
        modified_contact_image[i+1,j] = 0
        modified_contact_image[i+1,j+1] = 0

    # force_imageはcontact(01)との掛け算で作成する
    modified_force_image *= modified_contact_image

    return modified_initial_geometry_image,\
           modified_initial_depth_image,\
           modified_geometry_image,\
           modified_depth_image,\
           modified_contact_image,\
           modified_force_image,\
           modified_original_rgb_image,\
           modified_original_depth_image


# データセットの読み込み
force_images = np.load('original_dataset/force_images.npy')
initial_geometry_images = np.load('original_dataset/initial_geometry_images.npy')
initial_depth_images = np.load('original_dataset/initial_depth_images.npy')
geometry_images = np.load('original_dataset/geometry_images.npy')
depth_images = np.load('original_dataset/depth_images.npy')
contact_images = np.load('original_dataset/contact_images.npy')
force_images = np.load('original_dataset/force_images.npy')

original_rgb_images = np.load('original_dataset/original_rgb_images.npy')
original_rgb_images = original_rgb_images.astype(np.uint8)
original_depth_images = np.load('original_dataset/original_depth_images.npy')

# オリジナルのデータセットを0次元でランダムにシャッフルする
shuffled_indices = np.random.permutation(initial_geometry_images.shape[0])
train_indices = shuffled_indices[0:int(len(shuffled_indices)*0.9)]
test_indices = shuffled_indices[int(len(shuffled_indices)*0.9):]


# 拡張前の学習データ
initial_geometry_tr_images = initial_geometry_images[train_indices]
initial_depth_tr_images = initial_depth_images[train_indices]
geometry_tr_images = geometry_images[train_indices]
depth_tr_images = depth_images[train_indices]
contact_tr_images = contact_images[train_indices]
force_tr_images = force_images[train_indices]

original_rgb_tr_images = original_rgb_images[train_indices]
original_depth_tr_images = original_depth_images[train_indices]

# 各画像に対して10回ずつデータ拡張を行う
count = 0
for i in range(initial_geometry_tr_images.shape[0]):
    print(i,count)
    # オリジナルは最初に保存する
    initial_geometry_image = initial_geometry_tr_images[i].astype(np.float32)
    initial_depth_image = initial_depth_tr_images[i].astype(np.float32)
    geometry_image = geometry_tr_images[i].astype(np.float32)
    depth_image = depth_tr_images[i].astype(np.float32)
    contact_image = contact_tr_images[i].astype(np.float32)
    force_image = force_tr_images[i].astype(np.float32)

    original_rgb_image = original_rgb_tr_images[i].astype(np.uint8)
    original_depth_image = original_depth_tr_images[i].astype(np.int16)

    np.save('dataset/train/initial_geometry_images/' + str(count) + '.npy', initial_geometry_image)
    np.save('dataset/train/initial_depth_images/' + str(count) + '.npy', initial_depth_image)
    np.save('dataset/train/geometry_images/' + str(count) + '.npy', geometry_image)
    np.save('dataset/train/depth_images/' + str(count) + '.npy', depth_image)
    np.save('dataset/train/contact_images/' + str(count) + '.npy', contact_image)
    np.save('dataset/train/force_images/' + str(count) + '.npy', force_image)

    np.save('dataset/train/original_rgb_images/' + str(count) + '.npy', original_rgb_image)
    np.save('dataset/train/original_depth_images/' + str(count) + '.npy', original_depth_image)

    count += 1

    for _ in range(10):
        initial_geometry_image,initial_depth_image,geometry_image,depth_image,contact_image,force_image,original_rgb_image,original_depth_image = \
            randam_expand_and_slide(initial_geometry_tr_images[i],initial_depth_tr_images[i],geometry_tr_images[i],depth_tr_images[i],contact_tr_images[i],force_tr_images[i],original_rgb_tr_images[i],original_depth_tr_images[i])

        initial_geometry_image = initial_geometry_image.astype(np.float32)
        initial_depth_image = initial_depth_image.astype(np.float32)
        geometry_image = geometry_image.astype(np.float32)
        depth_image = depth_image.astype(np.float32)
        contact_image = contact_image.astype(np.float32)
        force_image = force_image.astype(np.float32)

        original_rgb_image = original_rgb_image.astype(np.uint8)
        original_depth_image = original_depth_image.astype(np.int16)

        np.save('dataset/train/initial_geometry_images/' + str(count) + '.npy', initial_geometry_image)
        np.save('dataset/train/initial_depth_images/' + str(count) + '.npy', initial_depth_image)
        np.save('dataset/train/geometry_images/' + str(count) + '.npy', geometry_image)
        np.save('dataset/train/depth_images/' + str(count) + '.npy', depth_image)
        np.save('dataset/train/contact_images/' + str(count) + '.npy', contact_image)
        np.save('dataset/train/force_images/' + str(count) + '.npy', force_image)

        np.save('dataset/train/original_rgb_images/' + str(count) + '.npy', original_rgb_image)
        np.save('dataset/train/original_depth_images/' + str(count) + '.npy', original_depth_image)

        count += 1

        print(i, count)


# 拡張前のテストデータ
initial_geometry_va_images = initial_geometry_images[test_indices]
initial_depth_va_images = initial_depth_images[test_indices]
geometry_va_images = geometry_images[test_indices]
depth_va_images = depth_images[test_indices]
contact_va_images = contact_images[test_indices]
force_va_images = force_images[test_indices]

original_rgb_va_images = original_rgb_images[test_indices]
original_depth_va_images = original_depth_images[test_indices]

# 各画像に対して10回ずつデータ拡張を行う
count = 0
for i in range(initial_geometry_va_images.shape[0]):
    print(i, count)
    # オリジナルは最初に保存する
    initial_geometry_image = initial_geometry_va_images[i].astype(np.float32)
    initial_depth_image = initial_depth_va_images[i].astype(np.float32)
    geometry_image = geometry_va_images[i].astype(np.float32)
    depth_image = depth_va_images[i].astype(np.float32)
    contact_image = contact_va_images[i].astype(np.float32)
    force_image = force_va_images[i].astype(np.float32)

    original_rgb_image = original_rgb_va_images[i].astype(np.uint8)
    original_depth_image = original_depth_va_images[i].astype(np.int16)

    np.save('dataset/test/initial_geometry_images/' + str(count) + '.npy', initial_geometry_image)
    np.save('dataset/test/initial_depth_images/' + str(count) + '.npy', initial_depth_image)
    np.save('dataset/test/geometry_images/' + str(count) + '.npy', geometry_image)
    np.save('dataset/test/depth_images/' + str(count) + '.npy', depth_image)
    np.save('dataset/test/contact_images/' + str(count) + '.npy', contact_image)
    np.save('dataset/test/force_images/' + str(count) + '.npy', force_image)

    np.save('dataset/test/original_rgb_images/' + str(count) + '.npy', original_rgb_image)
    np.save('dataset/test/original_depth_images/' + str(count) + '.npy', original_depth_image)

    count += 1

    for _ in range(10):
        initial_geometry_image, initial_depth_image, geometry_image, depth_image, contact_image, force_image, original_rgb_image, original_depth_image = \
            randam_expand_and_slide(initial_geometry_va_images[i], initial_depth_va_images[i], geometry_va_images[i], depth_va_images[i], contact_va_images[i], force_va_images[i], original_rgb_va_images[i], original_depth_va_images[i])

        initial_geometry_image = initial_geometry_image.astype(np.float32)
        initial_depth_image = initial_depth_image.astype(np.float32)
        geometry_image = geometry_image.astype(np.float32)
        depth_image = depth_image.astype(np.float32)
        contact_image = contact_image.astype(np.float32)
        force_image = force_image.astype(np.float32)

        original_rgb_image = original_rgb_image.astype(np.uint8)
        original_depth_image = original_depth_image.astype(np.int16)

        np.save('dataset/test/initial_geometry_images/' + str(count) + '.npy', initial_geometry_image)
        np.save('dataset/test/initial_depth_images/' + str(count) + '.npy', initial_depth_image)
        np.save('dataset/test/geometry_images/' + str(count) + '.npy', geometry_image)
        np.save('dataset/test/depth_images/' + str(count) + '.npy', depth_image)
        np.save('dataset/test/contact_images/' + str(count) + '.npy', contact_image)
        np.save('dataset/test/force_images/' + str(count) + '.npy', force_image)

        np.save('dataset/test/original_rgb_images/' + str(count) + '.npy', original_rgb_image)
        np.save('dataset/test/original_depth_images/' + str(count) + '.npy', original_depth_image)

        count += 1

        print(i, count)

# 一つのnpyファイルとしてまとめる方法
'''
import numpy as np
from scipy import ndimage
import math
import os

def determine_offset(scale):
    LR = np.random.uniform(-320*(scale-1), 320*(scale-1))
    # 絶対値が小さくなるように切り捨て
    if LR>0:
        LR = math.floor(LR)
    else:
        LR = math.ceil(LR)

    UD = np.random.uniform(-288*(scale-1), 288*(scale-1))
    # 絶対値が小さくなるように切り捨て
    if UD>0:
        UD = math.floor(UD)
    else:
        UD = math.ceil(UD)

    return (LR, UD)

def randam_expand_and_slide(
        initial_geometry_image,
        initial_depth_image,
        geometry_image,
        depth_image,
        contact_image,
        force_image,
        original_rgb_image,
        original_depth_image):

    # rgbを各チャンネルに分割して処理して最後に3チャンネルに結合する
    original_r_image = original_rgb_image[:,:,0]
    original_g_image = original_rgb_image[:,:,1]
    original_b_image = original_rgb_image[:,:,2]

    # 画像をランダムな倍数で拡大する
    random_scale = np.random.uniform(1, 1.2)

    modified_initial_geometry_image = ndimage.zoom(initial_geometry_image, random_scale, order=0)
    modified_initial_depth_image = ndimage.zoom(initial_depth_image, random_scale, order=0)
    modified_geometry_image = ndimage.zoom(geometry_image, random_scale, order=0)
    modified_depth_image = ndimage.zoom(depth_image, random_scale, order=0)
    modified_contact_image = ndimage.zoom(contact_image, random_scale, order=0)
    modified_force_image = ndimage.zoom(force_image, random_scale, order=0)

    modified_original_r_image = ndimage.zoom(original_r_image, random_scale, order=0)
    modified_original_g_image = ndimage.zoom(original_g_image, random_scale, order=0)
    modified_original_b_image = ndimage.zoom(original_b_image, random_scale, order=0)
    modified_original_depth_image = ndimage.zoom(original_depth_image, random_scale, order=0)

    # 画像をランダムなオフセットでスライドさせる
    offset = determine_offset(random_scale)

    # 拡大した画像から中心からランダムなオフセットの分だけずらして640×576ピクセルを抽出する
    start_x = (modified_initial_geometry_image.shape[1]-640)//2
    start_y = (modified_initial_geometry_image.shape[0]-576)//2

    modified_initial_geometry_image = modified_initial_geometry_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]
    modified_initial_depth_image = modified_initial_depth_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]
    modified_geometry_image = modified_geometry_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]
    modified_depth_image = modified_depth_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]
    modified_contact_image = modified_contact_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]
    modified_force_image = modified_force_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]

    modified_original_r_image = modified_original_r_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]
    modified_original_g_image = modified_original_g_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]
    modified_original_b_image = modified_original_b_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]
    modified_original_rgb_image = np.stack([modified_original_r_image, modified_original_g_image, modified_original_b_image], axis=2)
    modified_original_depth_image = modified_original_depth_image[(start_y+offset[1]):(start_y+offset[1])+576, (start_x+offset[0]):(start_x+offset[0])+640]

    return modified_initial_geometry_image,\
           modified_initial_depth_image,\
           modified_geometry_image,\
           modified_depth_image,\
           modified_contact_image,\
           modified_force_image,\
           modified_original_rgb_image,\
           modified_original_depth_image


# データセットの読み込み
initial_geometry_images = np.load('original_dataset/initial_geometry_images.npy')
initial_depth_images = np.load('original_dataset/initial_depth_images.npy')
geometry_images = np.load('original_dataset/geometry_images.npy')
depth_images = np.load('original_dataset/depth_images.npy')
contact_images = np.load('original_dataset/contact_images.npy')
force_images = np.load('original_dataset/force_images.npy')

original_rgb_images = np.load('original_dataset/original_rgb_images.npy')
original_rgb_images = original_rgb_images.astype(np.uint8)
original_depth_images = np.load('original_dataset/original_depth_images.npy')

# オリジナルのデータセットを0次元でランダムにシャッフルする
shuffled_indices = np.random.permutation(initial_geometry_images.shape[0])
train_indices = shuffled_indices[0:int(len(shuffled_indices)*0.9)]
test_indices = shuffled_indices[int(len(shuffled_indices)*0.9):]


# 拡張前の学習データ
initial_geometry_tr_images = initial_geometry_images[train_indices]
initial_depth_tr_images = initial_depth_images[train_indices]
geometry_tr_images = geometry_images[train_indices]
depth_tr_images = depth_images[train_indices]
contact_tr_images = contact_images[train_indices]
force_tr_images = force_images[train_indices]

original_rgb_tr_images = original_rgb_images[train_indices]
original_depth_tr_images = original_depth_images[train_indices]

# 拡張後の学習データの保存先
initial_geometry_tr_images_output = np.empty((0,576,640))
initial_depth_tr_images_output = np.empty((0,576,640))
geometry_tr_images_output = np.empty((0,576,640))
depth_tr_images_output = np.empty((0,576,640))
contact_tr_images_output = np.empty((0,576,640))
force_tr_images_output = np.empty((0,576,640))

original_rgb_tr_images_output = np.empty((0,576,640,3), dtype=np.uint8)
original_depth_tr_images_output = np.empty((0,576,640))

# 保存するときのファイル名
output_files = ['contact_images',
                'depth_images',
                'force_images',
                'geometry_images',
                'initial_depth_images',
                'initial_geometry_images',
                'original_depth_images',
                'original_rgb_images']

# 各画像に対して10回ずつデータ拡張を行う
count = 0
for i in range(initial_geometry_tr_images.shape[0]):
    print(i)
    # オリジナルは最初に保存する
    initial_geometry_tr_images_output = np.concatenate((initial_geometry_tr_images_output,initial_geometry_tr_images[i][np.newaxis,:]), axis=0).astype(np.float32)
    initial_depth_tr_images_output = np.concatenate((initial_depth_tr_images_output,initial_depth_tr_images[i][np.newaxis,:]), axis=0).astype(np.float32)
    geometry_tr_images_output = np.concatenate((geometry_tr_images_output,geometry_tr_images[i][np.newaxis,:]), axis=0).astype(np.float32)
    depth_tr_images_output = np.concatenate((depth_tr_images_output,depth_tr_images[i][np.newaxis,:]), axis=0).astype(np.float32)
    contact_tr_images_output = np.concatenate((contact_tr_images_output,contact_tr_images[i][np.newaxis,:]), axis=0).astype(np.float32)
    force_tr_images_output = np.concatenate((force_tr_images_output,force_tr_images[i][np.newaxis,:]), axis=0).astype(np.float32)

    original_rgb_tr_images_output = np.concatenate((original_rgb_tr_images_output,original_rgb_tr_images[i][np.newaxis,:]), axis=0).astype(np.uint8)
    original_depth_tr_images_output = np.concatenate((original_depth_tr_images_output,original_depth_tr_images[i][np.newaxis,:]), axis=0).astype(np.float32)

    for _ in range(10):
        initial_geometry_image,initial_depth_image,geometry_image,depth_image,contact_image,force_image,original_rgb_image,original_depth_image = \
            randam_expand_and_slide(initial_geometry_tr_images[i],initial_depth_tr_images[i],geometry_tr_images[i],depth_tr_images[i],contact_tr_images[i],force_tr_images[i],original_rgb_tr_images[i],original_depth_tr_images[i])

        initial_geometry_tr_images_output = np.concatenate((initial_geometry_tr_images_output, initial_geometry_image[np.newaxis, :]), axis=0).astype(np.float32)
        initial_depth_tr_images_output = np.concatenate((initial_depth_tr_images_output, initial_depth_image[np.newaxis, :]), axis=0).astype(np.float32)
        geometry_tr_images_output = np.concatenate((geometry_tr_images_output, geometry_image[np.newaxis, :]), axis=0).astype(np.float32)
        depth_tr_images_output = np.concatenate((depth_tr_images_output, depth_image[np.newaxis, :]), axis=0).astype(np.float32)
        contact_tr_images_output = np.concatenate((contact_tr_images_output, contact_image[np.newaxis, :]), axis=0).astype(np.float32)
        force_tr_images_output = np.concatenate((force_tr_images_output, force_image[np.newaxis, :]), axis=0).astype(np.float32)

        original_rgb_image = original_rgb_image.astype(np.uint8)
        original_rgb_tr_images_output = np.concatenate((original_rgb_tr_images_output, original_rgb_image[np.newaxis,:]), axis=0).astype(np.uint8)
        original_depth_tr_images_output = np.concatenate((original_depth_tr_images_output, original_depth_image[np.newaxis, :]), axis=0).astype(np.float32)

    if i % 30 == 1:
        print('データ書き出し完了')

        # データ書き出し(0→1→10→11‥→2→21とならないように2桁スタートにする)
        np.save('dataset/train/initial_geometry_images_'+str(count+10)+'.npy', initial_geometry_tr_images_output)
        np.save('dataset/train/initial_depth_images_'+str(count+10)+'.npy', initial_depth_tr_images_output)
        np.save('dataset/train/geometry_images_'+str(count+10)+'.npy', geometry_tr_images_output)
        np.save('dataset/train/depth_images_'+str(count+10)+'.npy', depth_tr_images_output)
        np.save('dataset/train/contact_images_'+str(count+10)+'.npy', contact_tr_images_output)
        np.save('dataset/train/force_images_'+str(count+10)+'.npy', force_tr_images_output)
        np.save('dataset/train/original_rgb_images_'+str(count+10)+'.npy', original_rgb_tr_images_output)
        np.save('dataset/train/original_depth_images_'+str(count+10)+'.npy', original_depth_tr_images_output)

        # 拡張後の学習データの保存先をリセット
        initial_geometry_tr_images_output = np.empty((0,576,640))
        initial_depth_tr_images_output = np.empty((0,576,640))
        geometry_tr_images_output = np.empty((0,576,640))
        depth_tr_images_output = np.empty((0,576,640))
        contact_tr_images_output = np.empty((0,576,640))
        force_tr_images_output = np.empty((0,576,640))

        original_rgb_tr_images_output = np.empty((0,576,640,3), dtype=np.uint8)
        original_depth_tr_images_output = np.empty((0,576,640))

        count += 1

else:
    # データ書き出し(0→1→10→11‥→2→21とならないように2桁スタートにする)
    np.save('dataset/train/initial_geometry_images_'+str(count+10)+'.npy', initial_geometry_tr_images_output)
    np.save('dataset/train/initial_depth_images_'+str(count+10)+'.npy', initial_depth_tr_images_output)
    np.save('dataset/train/geometry_images_'+str(count+10)+'.npy', geometry_tr_images_output)
    np.save('dataset/train/depth_images_'+str(count+10)+'.npy', depth_tr_images_output)
    np.save('dataset/train/contact_images_'+str(count+10)+'.npy', contact_tr_images_output)
    np.save('dataset/train/force_images_'+str(count+10)+'.npy', force_tr_images_output)
    np.save('dataset/train/original_rgb_images_'+str(count+10)+'.npy', original_rgb_tr_images_output)
    np.save('dataset/train/original_depth_images_'+str(count+10)+'.npy', original_depth_tr_images_output)

# メモリから消去
del initial_geometry_tr_images_output
del initial_depth_tr_images_output
del geometry_tr_images_output
del depth_tr_images_output
del contact_tr_images_output
del force_tr_images_output
del original_rgb_tr_images_output
del original_depth_tr_images_output

# 分割したnpyファイルを結合する
files = os.listdir('dataset/train')

# 各ファイルに対して存在するcount+1個のデータを1つのファイルに結合する
for i in range(int(len(output_files))):
    print(str(i)+'周目開始')

    # rgb(index=7)のときは3チャンネルを用意する
    output_data = np.empty((0,576,640))
    if i == 7:
        output_data = np.empty((0,576,640,3), dtype=np.uint8)

    # count+1個のデータを順番に結合して大きくしていく
    for j in range(count+1):
        tmp = np.load('dataset/train/'+files[(count+1)*i+j])
        output_data = np.concatenate((output_data, tmp), axis=0)
        os.remove('dataset/train/'+files[(count+1)*i+j])
    else:
        np.save('dataset/train/'+output_files[i]+'.npy',output_data)
print('データ結合完了')


# 拡張前のテストデータ
initial_geometry_va_images = initial_geometry_images[test_indices]
initial_depth_va_images = initial_depth_images[test_indices]
geometry_va_images = geometry_images[test_indices]
depth_va_images = depth_images[test_indices]
contact_va_images = contact_images[test_indices]
force_va_images = force_images[test_indices]

original_rgb_va_images = original_rgb_images[test_indices]
original_depth_va_images = original_depth_images[test_indices]

# 拡張後のテストデータの保存先
initial_geometry_va_images_output = np.empty((0,576,640))
initial_depth_va_images_output = np.empty((0,576,640))
geometry_va_images_output = np.empty((0,576,640))
depth_va_images_output = np.empty((0,576,640))
contact_va_images_output = np.empty((0,576,640))
force_va_images_output = np.empty((0,576,640))

original_rgb_va_images_output = np.empty((0,576,640,3), dtype=np.uint8)
original_depth_va_images_output = np.empty((0,576,640))

# 各画像に対して10回ずつデータ拡張を行う
count = 0
for i in range(initial_geometry_va_images.shape[0]):
    print(i)
    # オリジナルは最初に保存する
    initial_geometry_va_images_output = np.concatenate((initial_geometry_va_images_output,initial_geometry_va_images[i][np.newaxis,:]), axis=0).astype(np.float32)
    initial_depth_va_images_output = np.concatenate((initial_depth_va_images_output,initial_depth_va_images[i][np.newaxis,:]), axis=0).astype(np.float32)
    geometry_va_images_output = np.concatenate((geometry_va_images_output,geometry_va_images[i][np.newaxis,:]), axis=0).astype(np.float32)
    depth_va_images_output = np.concatenate((depth_va_images_output,depth_va_images[i][np.newaxis,:]), axis=0).astype(np.float32)
    contact_va_images_output = np.concatenate((contact_va_images_output,contact_va_images[i][np.newaxis,:]), axis=0).astype(np.float32)
    force_va_images_output = np.concatenate((force_va_images_output,force_va_images[i][np.newaxis,:]), axis=0).astype(np.float32)

    original_rgb_va_images_output = np.concatenate((original_rgb_va_images_output,original_rgb_va_images[i][np.newaxis,:]), axis=0).astype(np.uint8)
    original_depth_va_images_output = np.concatenate((original_depth_va_images_output,original_depth_va_images[i][np.newaxis,:]), axis=0).astype(np.float32)

    for _ in range(10):
        initial_geometry_image,initial_depth_image,geometry_image,depth_image,contact_image,force_image,original_rgb_image,original_depth_image = \
            randam_expand_and_slide(initial_geometry_va_images[i],initial_depth_va_images[i],geometry_va_images[i],depth_va_images[i],contact_va_images[i],force_va_images[i],original_rgb_va_images[i],original_depth_va_images[i])

        initial_geometry_va_images_output = np.concatenate((initial_geometry_va_images_output, initial_geometry_image[np.newaxis, :]), axis=0).astype(np.float32)
        initial_depth_va_images_output = np.concatenate((initial_depth_va_images_output, initial_depth_image[np.newaxis, :]), axis=0).astype(np.float32)
        geometry_va_images_output = np.concatenate((geometry_va_images_output, geometry_image[np.newaxis, :]), axis=0).astype(np.float32)
        depth_va_images_output = np.concatenate((depth_va_images_output, depth_image[np.newaxis, :]), axis=0).astype(np.float32)
        contact_va_images_output = np.concatenate((contact_va_images_output, contact_image[np.newaxis, :]), axis=0).astype(np.float32)
        force_va_images_output = np.concatenate((force_va_images_output, force_image[np.newaxis, :]), axis=0).astype(np.float32)

        original_rgb_va_images_output = np.concatenate((original_rgb_va_images_output, original_rgb_image[np.newaxis,:]), axis=0).astype(np.uint8)
        original_depth_va_images_output = np.concatenate((original_depth_va_images_output, original_depth_image[np.newaxis, :]), axis=0).astype(np.float32)

    if i % 30 == 1:
        print('データ書き出し完了')

        # データ書き出し(0→1→10→11‥→2→21とならないように2桁スタートにする)
        np.save('dataset/test/initial_geometry_images_'+str(count+10)+'.npy', initial_geometry_va_images_output)
        np.save('dataset/test/initial_depth_images_'+str(count+10)+'.npy', initial_depth_va_images_output)
        np.save('dataset/test/geometry_images_'+str(count+10)+'.npy', geometry_va_images_output)
        np.save('dataset/test/depth_images_'+str(count+10)+'.npy', depth_va_images_output)
        np.save('dataset/test/contact_images_'+str(count+10)+'.npy', contact_va_images_output)
        np.save('dataset/test/force_images_'+str(count+10)+'.npy', force_va_images_output)
        np.save('dataset/test/original_rgb_images_'+str(count+10)+'.npy', original_rgb_va_images_output)
        np.save('dataset/test/original_depth_images_'+str(count+10)+'.npy', original_depth_va_images_output)

        # 拡張後の学習データの保存先をリセット
        initial_geometry_va_images_output = np.empty((0,576,640))
        initial_depth_va_images_output = np.empty((0,576,640))
        geometry_va_images_output = np.empty((0,576,640))
        depth_va_images_output = np.empty((0,576,640))
        contact_va_images_output = np.empty((0,576,640))
        force_va_images_output = np.empty((0,576,640))

        original_rgb_va_images_output = np.empty((0,576,640,3), dtype=np.uint8)
        original_depth_va_images_output = np.empty((0,576,640))

        count += 1

else:
    # データ書き出し(0→1→10→11‥→2→21とならないように2桁スタートにする)
    np.save('dataset/test/initial_geometry_images_'+str(count+10)+'.npy', initial_geometry_va_images_output)
    np.save('dataset/test/initial_depth_images_'+str(count+10)+'.npy', initial_depth_va_images_output)
    np.save('dataset/test/geometry_images_'+str(count+10)+'.npy', geometry_va_images_output)
    np.save('dataset/test/depth_images_'+str(count+10)+'.npy', depth_va_images_output)
    np.save('dataset/test/contact_images_'+str(count+10)+'.npy', contact_va_images_output)
    np.save('dataset/test/force_images_'+str(count+10)+'.npy', force_va_images_output)
    np.save('dataset/test/original_rgb_images_'+str(count+10)+'.npy', original_rgb_va_images_output)
    np.save('dataset/test/original_depth_images_'+str(count+10)+'.npy', original_depth_va_images_output)

# メモリから消去
del initial_geometry_va_images_output
del initial_depth_va_images_output
del geometry_va_images_output
del depth_va_images_output
del contact_va_images_output
del force_va_images_output
del original_rgb_va_images_output
del original_depth_va_images_output

# 分割したnpyファイルを結合する
files = os.listdir('dataset/test')

# 各ファイルに対して存在するcount+1個のデータを1つのファイルに結合する
for i in range(int(len(output_files))):
    print(str(i)+'周目開始')

    # rgb(index=7)のときは3チャンネルを用意する
    output_data = np.empty((0,576,640))
    if i == 7:
        output_data = np.empty((0,576,640,3), dtype=np.uint8)

    # count+1個のデータを順番に結合して大きくしていく
    for j in range(count+1):
        tmp = np.load('dataset/test/'+files[(count+1)*i+j])
        output_data = np.concatenate((output_data, tmp), axis=0)
        os.remove('dataset/test/'+files[(count+1)*i+j])
    else:
        np.save('dataset/test/'+output_files[i]+'.npy',output_data)
print('データ結合完了')
'''