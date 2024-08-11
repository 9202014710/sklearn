import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt

mac2id = dict()
onlinetimes = []
f = open(r'.\sklearn\test_txt\online_time.txt', encoding='utf-8')
for line in f:
    mac = line.split(',')[2]                        #读取每条数据中的MAC地址
    # print(mac)
    onlinetime = int(line.split(',')[6])            #上网时长
    # print(onlinetime)
    starttime = int(line.split(',')[4].split(' ')[1].split(':')[0])          #开始上网时间（小时）
    # print(line.split(',')[4])
    # print(starttime)
    if mac not in mac2id:
        mac2id[mac] = len(onlinetimes)              #mac2id是一个字典：key是mac地址 value是对应mac地址的上网时长以及开始上网时间
        onlinetimes.append((starttime, onlinetime))
    else:
        onlinetimes[mac2id[mac]] = [(starttime, onlinetime)]

print(mac2id)
print(onlinetimes)

real_X = np.array(onlinetimes).reshape((-1, 2))
# print(real_X)

# X = real_X[:, 0:1]          #取上网开始时间（小时）
X = np.log(1+real_X[:, 1:])	  #取上网时长聚类
print(X)

#db = skc.DBSCAN(eps=0.01, min_samples=20,metric='euclidean').fit(X)   
db = skc.DBSCAN(eps=0.14, min_samples=10,metric='euclidean').fit(X)            #调用DBSCAN方法进行训练，主要参数eps:俩个样本被看作邻居节点的最大距离，min_samples:簇的样本数，metric:距离计算方式，labels为每个数据的簇标签
labels = db.labels_

print('Labels:')                                            #打印数据被记上的标签，计算标签为-1，即噪声数据的比例
print(labels)
raito = len(labels[labels[:] == -1]) / len(labels)
print('Noise raito:', format(raito, '.2%'))

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)                    #计算簇的个数并打印，评价聚类效果

print('Estimated number of clusters: %d' % n_clusters_)                         #簇的个数
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))    #聚类效果评价指标

for i in range(n_clusters_):                        #打印各簇标号以及各簇内数据
    print('Cluster ', i, ':')
    print(list(X[labels == i].flatten()))

plt.hist(X, 24)
plt.show()
