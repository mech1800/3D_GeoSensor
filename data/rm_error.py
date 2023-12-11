import numpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

# クリックの位置で振り分けるコード
print('右半分○ 左半分×')

index = []

geometry_images = np.load('original_dataset/geometry_images.npy')

# クリック時の処理
def on_mouse_click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append((x,y))

# for i in range(20):
for i in range(geometry_images.shape[0]):
    image = cv2.resize(geometry_images[i], (800, 800))

    cv2.namedWindow('geometry')
    cv2.imshow('geometry', image)

    coords = []
    cv2.setMouseCallback('geometry', on_mouse_click)

    # 'q'が押されるまで待機
    while True:
        cv2.imshow('geometry', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if coords == []:
        continue
    if coords[-1][0] > 400:
        index.append(i)
        print('indexを追加しました')

with open('original_dataset/index.pickle','wb') as file:
    pickle.dump(index,file)

geometry_images = geometry_images[index]
np.save('original_dataset/geometry_images.npy',geometry_images)
del geometry_images

initial_geometry_images = np.load('original_dataset/initial_geometry_images.npy')
initial_geometry_images = initial_geometry_images[index]
np.save('original_dataset/initial_geometry_images.npy', initial_geometry_images)
del initial_geometry_images

depth_images = np.load('original_dataset/depth_images.npy')
depth_images = depth_images[index]
np.save('original_dataset/depth_images.npy', depth_images)
del depth_images

initial_depth_images = np.load('original_dataset/initial_depth_images.npy')
initial_depth_images = initial_depth_images[index]
np.save('original_dataset/initial_depth_images.npy', initial_depth_images)
del initial_depth_images

contact_images = np.load('original_dataset/contact_images.npy')
contact_images = contact_images[index]
np.save('original_dataset/contact_images.npy', contact_images)
del contact_images

force_images = np.load('original_dataset/force_images.npy')
force_images = force_images[index]
np.save('original_dataset/force_images.npy', force_images)
del force_images

original_depth_images = np.load('original_dataset/original_depth_images.npy')
original_depth_images = original_depth_images[index]
np.save('original_dataset/original_depth_images.npy', original_depth_images)
del original_depth_images

original_rgb_images = np.load('original_dataset/original_rgb_images.npy')
original_rgb_images = original_rgb_images[index]
np.save('original_dataset/original_rgb_images.npy', original_rgb_images)
del original_rgb_images