#模块导入
import pandas as pd
import numpy as np  

from sklearn import impute
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
#主函数-数据准备
def load_datasets(feature_paths, label_paths):
    feature = np.ndarray(shape=(0,41))
    label = np.ndarray(shape=(0,1))
    for file in feature_paths:
        df = pd.read_table(file, delimiter=',', na_values='?', header=None)
        imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(df)
        df = imp.transform(df)
        feature = np.concatenate((feature, df))
#数据导入函数    
    for file in label_paths:
        df = pd.read_table(file, header=None)
        label = np.concatenate((label, df))
        
    label = np.ravel(label)
    return feature, label
#主函数-数据准备
if __name__ == '__main__':
    ''' 数据路径 '''
    featurePaths = ['sklearn/A/A.feature','sklearn/B/B.feature','sklearn/C/C.feature','sklearn/D/D.feature','sklearn/E/E.feature']
    labelPaths = ['sklearn/A/A.label','sklearn/B/B.label','sklearn/C/C.label','sklearn/D/D.label','sklearn/E/E.label']
    ''' 读入数据  '''
    x_train,y_train = load_datasets(featurePaths[:4],labelPaths[:4])
    x_test,y_test = load_datasets(featurePaths[4:],labelPaths[4:])
    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size = 0.1)
#主函数-knn    
    print('Start training knn')
    knn = KNeighborsClassifier().fit(x_train, y_train)
    print('Training done')
    answer_knn = knn.predict(x_test)
    print('Prediction done')
#主函数-决策树    
    print('Start training DT')
    dt = DecisionTreeClassifier().fit(x_train, y_train)
    print('Training done')
    answer_dt = dt.predict(x_test)
    print('Prediction done')
#主函数-贝叶斯    
    print('Start training Bayes')
    gnb = GaussianNB().fit(x_train, y_train)
    print('Training done')
    answer_gnb = gnb.predict(x_test)
    print('Prediction done')
#主函数-分类结果分析，计算准确率和召回率    
    print('\n\nThe classification report for knn:')
    print(classification_report(y_test, answer_knn))
    print('\n\nThe classification report for DT:')
    print(classification_report(y_test, answer_dt))
    print('\n\nThe classification report for Bayes:')
    print(classification_report(y_test, answer_gnb))
