import cv2
import numpy as np

# フィルタ作成用の画像を読み込んでグレースケールにする
bgr_image = cv2.imread('mask_1.jpg')
gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

# 白い部分を0、その他を1とすることで乗算フィルタとする
gray_image[gray_image<250] = 1
gray_image[gray_image>=250] = 0

# numpyのnp.float32として保存
gray_image = gray_image.astype(np.float32)
np.save('mask_1.npy', gray_image)


# フィルタ作成用の画像を読み込んでグレースケールにする
bgr_image = cv2.imread('mask_2.jpg')
gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

# 白い部分を0、その他を1とすることで乗算フィルタとする
gray_image[gray_image<250] = 1
gray_image[gray_image>=250] = 0

# numpyのnp.float32として保存
gray_image = gray_image.astype(np.float32)
np.save('mask_2.npy', gray_image)