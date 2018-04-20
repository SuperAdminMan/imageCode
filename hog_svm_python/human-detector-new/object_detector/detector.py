
import glob
import os
import cv2
import numpy as np
from skimage.transform import pyramid_gaussian
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
from sklearn.externals import joblib
import matplotlib.pyplot as plt

def sliding_window(image, window_size, step_size):
    '''
    滑动窗口检测。这个函数返回大小等于window_size的image patch，从左上角开始移动，移动步长为setp_size
    :param image:输入的图像
    :param window_size:窗口大小即patch大小
    :param step_size:移动步长
    :return:返回一个tuple(x, y, im_window)
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])


def detector(filename):
    image = cv2.imread(filename)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    min_window_size = (64, 128)
    step_size = (10, 10)
    downscale = 1.25    #金字塔的缩放比例

    clf = joblib.load(os.path.join('./../data/models', 'svm.pkl'))

    detections = []
    scale = 0
    #利用图像金字塔对图像进行缩放
    for im_scaled in pyramid_gaussian(image, downscale=downscale):
        if im_scaled.shape[0] < min_window_size[1] or im_scaled.shape[1] < min_window_size[0]:
            break
        for (x, y ,image_window) in sliding_window(im_scaled, min_window_size, step_size):
            if image_window.shape[0] != min_window_size[1] or image_window.shape[1] != min_window_size[0]:
                continue
            image_window = color.rgb2gray(image_window)
            image_feature = hog(image=image_window, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2))
            image_feature = image_feature.reshape(1, -1)
            pred = clf.predict(image_feature)

            if pred == 1:
                if clf.decision_function(image_feature) > 0.5:  #计算image_feature与分割超平面的函数距离
                    detections.append(
                        (int(x * (downscale ** scale)), int(y * (downscale ** scale)), clf.decision_function(image_feature),
                         int(min_window_size[0] * (downscale ** scale)),
                         int(min_window_size[1] * (downscale ** scale))))
        scale += 1

    clone = image.copy()

    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(image, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness=2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print("sc: ", sc)
    sc = np.array(sc)
    #使用非极大值抑制函数
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
    print("shape, ", pick.shape)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Raw Detection before NMS")
    plt.show()

    plt.axis("off")
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title("Final Detections after applying NMS")
    plt.show()

def test_folder(foldername):
    '''
    对foldername文件中的所有图片进行detector()处理
    :param foldername: 文件目录
    :return: 无
    '''
    filenames = glob.iglob(os.path.join(foldername, '*'))
    for filename in filenames:
        print(filename)
        detector(filename)


if __name__ == '__main__':
    foldername = './../data/test_image'
    test_folder(foldername)