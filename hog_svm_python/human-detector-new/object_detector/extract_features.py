
import os
import glob
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib


def extract_features(images_path, features_path, features_type):
    '''
    提取image_path目录下的图片特征，并保存到features_path目录下。
    :param images_path: 保存的图片目录
    :param features_path: 特征保存目录
    :param features_type: 特征类型：hog特征
    :return: 无返回值
    '''
    #如果保存特征的路径不存在，进行创建
    if not os.path.isdir(features_path):
        os.makedirs(features_path)

    #依次取出iamge_path目录下的图片，将每个图片的路径保存到image_path中
    for image_path in glob.glob(os.path.join(images_path, '*')):
        image = imread(image_path, as_grey=True)    #灰度图
        if features_type == 'HOG':
            #计算hog特征
            image_feature = hog(image=image, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2))
        #对该图片的特征进行命名，方便保存
        image_feature_name = os.path.split((image_path))[1].split('.')[0] + '.feat'
        #该图片特征的保存目录
        image_feature_path = os.path.join(features_path, image_feature_name)
        #将该图片的特征保存到希望保存的目录中
        joblib.dump(image_feature, image_feature_path)

if __name__ == '__main__':
    features_type = 'HOG'
    positive_features_path = './../data/features/positive_features'
    negative_features_path = './../data/features/negative_features'
    positive_images_path = './../data/images/pos_person'
    negative_images_path = './../data/images/neg_person'
    print('提取data/images/pos_person中图片特征，并保存到data/features/positive_features中！')
    extract_features(positive_images_path, positive_features_path, features_type)
    print('提取data/images/neg_person中图片特征，并保存到data/features/negative_features中！')
    extract_features(negative_images_path, negative_features_path, features_type)
    print('特征提取结束！')