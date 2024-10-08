# 使用PCA（主成分分析）对黛尾花（Iris）数据集进行降维

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# 加载数据集
data = load_iris()
y = data.target
X = data.data

# 初始化PCA模型并拟合数据
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_X)):
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])

# 绘制降维后的数据散点图
plt.scatter(red_x, red_y, c='r', marker='D')
plt.scatter(blue_x, blue_y, c='b')
plt.scatter(green_x, green_y, c='g', marker='x')
plt.show()
