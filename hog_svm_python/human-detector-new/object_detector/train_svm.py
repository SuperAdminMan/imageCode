
import glob
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import numpy as np

def train_svm(positive_features_path, negativate_features_path, model_path, classifer_type):
    '''
    训练分类器
    :param positive_features_path:    positive特征对应的目录
    :param negativate_features_path:     negative特征对应的目录
    :param model_path:    分类器保存的目录
    :param classifer_type:   分类器种类，为线性分类器
    :return:
    '''
    features_datas = []    #保存特征
    labels = []     #保存标签

    #循环加载positive特征
    for features_path in glob.glob(os.path.join(positive_features_path, '*.feat')):
        features_data = joblib.load(features_path)
        features_datas.append(features_data)
        labels.append(1)


    for features_path in glob.glob(os.path.join(negativate_features_path, '*feat')):
        features_data = joblib.load(features_path)
        features_datas.append(features_data)
        labels.append(0)

    print(np.array(features_datas).shape, len(labels))

    if classifer_type == 'LIN_SVM':
        classifer = LinearSVC()
        print('Training a Linear SVM Classifier')
        classifer.fit(features_datas, labels)

        #如果没有保存模型的路径，就创建该路径
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])

        model_path_name = os.path.join(model_path, 'svm.pkl')
        joblib.dump(classifer, model_path_name)
        print("Classifier saved to {}".format(model_path))

if __name__ == '__main__':
    positive_features_path = './../data/features/positive_features'
    negativate_features_path = './../data/features/negative_features'
    model_path = './../data/models/'
    classifer_type = 'LIN_SVM'
    train_svm(positive_features_path, negativate_features_path, model_path, classifer_type)
