import cv2
import numpy as np
from sklearn.cluster import KMeans
import collections

# maskが0の部分を白色[255,255,255]に置き換える
def attach_mask(bgr_image,mask):
    bgr_image[mask==0] = [255,255,255]
    return bgr_image

# カメラが計測出来ない黒色の部分([0,0,0]に近い部分)をバランスボール色に置き換える
def replace_black_pixel(bgr_image,bgr):
    black_pixel = (bgr_image <= [10,10,10]).all(axis=2)
    bgr_image[black_pixel] = bgr
    return bgr_image

# hsv画像からスーパーピクセルを求める(各ピクセルごとに何かしらのラベルが割り振られる)
def super_pixel(hsv_image):
    # スーパーピクセルを作成
    algorithm = cv2.ximgproc.SLIC
    region_size = 10   # 小さいほどスーパーピクセルのサイズが小さくなり細分化されたスーパーピクセルが生成されます(サイズを制御)
    ruler = 30   # 小さいほどスーパーピクセルの形状は細分化され細かいディテールが保持される可能性が高くなる(形状を制御)

    slic = cv2.ximgproc.createSuperpixelSLIC(image=hsv_image, algorithm=algorithm, region_size=region_size, ruler=float(ruler))

    # スーパーピクセルを計算
    min_element_size = 20   # このサイズ未満のスーパーピクセルは連結性の強制対象となる
    num_iterations = 10

    slic.iterate(num_iterations)
    slic.enforceLabelConnectivity(min_element_size)

    # スーパーピクセルのラベルを取得
    label = slic.getLabels()

    # スーパーピクセルの境界をbrg画像上に表示する（可視化部分）
    brg_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR_FULL)
    result = brg_image.copy()

    # 保存したいなら
    contour_mask = slic.getLabelContourMask(False)
    result[0 < contour_mask] = (0, 255, 255)
    cv2.imwrite('check/super_pixel_edge.jpg', result)

    return label


# スーパーピクセルをクラスタリングする(各ピクセルごとに何かしらのラベルが割り振られる)
def clustering(label, hsv_image):
    # 各スーパーピクセルの平均値を入れるリスト
    clustering_in = []

    # スーパーピクセルの各要素を平均値で置き換える
    mean_image = hsv_image.copy()

    for id in np.unique(label):
        target = np.where(label == id)

        h = [hsv_image[h, w, 0] for h, w in zip(target[0], target[1])]
        s = [hsv_image[h, w, 1] for h, w in zip(target[0], target[1])]
        v = [hsv_image[h, w, 2] for h, w in zip(target[0], target[1])]

        h_mean = np.mean(h)
        s_mean = np.mean(s)
        v_mean = np.mean(v)

        clustering_in.append([h_mean, s_mean, v_mean])

        # スーパーピクセルごとに平均値で置き換える
        for h, w in zip(target[0], target[1]):
            mean_image[h, w, 0] = h_mean
            mean_image[h, w, 1] = s_mean
            mean_image[h, w, 2] = v_mean

    # 平均値で置き換えたスーパーピクセルをbrg画像上に表示する（可視化部分）
    brg_image = cv2.cvtColor(mean_image, cv2.COLOR_HSV2BGR_FULL)
    # 保存したいなら
    cv2.imwrite('check/super_pixel_mean.jpg', brg_image)

    # clustering_in(各スーパーピクセルの平均[h,s,v])を使ってクラスタリング
    clustering_in = np.array(clustering_in)
    kmeans_result = KMeans(n_clusters=3, random_state=0).fit_predict(clustering_in)   # クラスタ数の指定は大事

    # スーパーピクセルのラベルをkmeansのラベルで置き換える
    clustering_label = cv2.cvtColor(brg_image, cv2.COLOR_RGB2GRAY)

    for i, id in enumerate(np.unique(label)):
        target = np.where(label == id)

        # ピクセルごとにクラスタidを割り振る
        for h, w in zip(target[0], target[1]):
            clustering_label[h, w] = kmeans_result[i]*50   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # 保存したいなら
    cv2.imwrite('check/kmeans.jpg', clustering_label)

    return clustering_label


def cluster_to_object(label):
    # 192,299のピクセルに存在するクラスタを物体ラベルとする
    object_id = label[384,299]
    object_label = np.where(label == object_id, 1, 0)
    cv2.imwrite('check/object_id.jpg', object_label)

    '''
    # 中心を1/2サイズで切り抜く
    half_label = label[int(label.shape[0]*1/4):int(label.shape[0]*3/4),int(label.shape[1]*1/4):int(label.shape[1]*3/4)]

    # 各クラスタ番号が何個あるか調べる
    label_1d = label.flatten()
    label_count = collections.Counter(label_1d)

    half_label_1d = half_label.flatten()
    half_label_count = collections.Counter(half_label_1d)

    # 増加率が大きいクラスタ番号を取得する
    tmp = -1
    for key in half_label_count:
        if (half_label_count[key]/len(half_label_1d)-label_count[key]/len(label_1d)) > tmp:
            object_id = key
            tmp = (half_label_count[key]/len(half_label_1d)-label_count[key]/len(label_1d))

    object_label = np.where(label == object_id, 1, 0)   # 50→1に変えた
    # 保存したいなら
    cv2.imwrite('check/object_id.jpg', object_label)
    '''

    return object_label


def Extract_object(bgr_image):
    # maskを使ってクラスタリングの精度を上げる
    mask_1 = np.load('../kinect_and_sensor/mask_1.npy')
    bgr_image[mask_1 == 0] = [255, 255, 255]

    # カメラが計測出来ない黒色の部分をバランスボール色に置き換える
    bgr_image = replace_black_pixel(bgr_image,bgr=[[64,41,37]])

    # hsv画像を用意する
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV_FULL)  # BGR-HSV変換

    # hsv画像をスーパーピクセルにする
    super_pixel_label = super_pixel(hsv_image)

    # スーパーピクセルをクラスタリングする
    clustering_label = clustering(super_pixel_label, hsv_image)

    # 物体を抽出する
    object_label = cluster_to_object(clustering_label)

    # maskを使ってクラスタリングの精度を上げる
    mask_1 = np.load('../kinect_and_sensor/mask_2.npy')
    object_label[mask_1 == 0] = 0

    return object_label