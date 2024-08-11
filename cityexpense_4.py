"""
数据介绍:现有1999年全国31个省份城镇居民家庭平均每人全年消费性支出的八个主要变量数据,
这八个变量分别是:食品、衣着、家庭设备用品及服务、医疗保健、交通和通讯、娱乐教育文化服务、居住以及杂项商品和服务。
利用已有数据,对31个省份进行聚类。

分类的组数不一样，则分类结果也会有所变化。
这里安装组数为3和4进行对比分析。
"""

import numpy as np
from sklearn.cluster import KMeans

def loadData(filePath):
    """加载数据"""
    with open(filePath, 'r', encoding='UTF-8') as fr:
        # 一次性读取整个文件，按行读取
        lines = fr.readlines()
        # 将读入的数据进行拆分，分s为数据和城市名
        retData = []
        retCityName = []
        for line in lines:
            # 去除字符串首尾的空格或者回车，并使用“，”进行分割
            items = line.strip().split(",")
            # 每一行的开头是城市名称
            retCityName.append(items[0])
            # 将数据组合成一个列表，并且强制转换类型为float浮点型
            retData.append([float(items[i]) for i in range(1, len(items))])
    return retData, retCityName

def process_data(input_file):
    # 使用读取数据，获取城市名和相关的数据
    data, cityName = loadData(input_file)
    # 创建指定簇数量KMeans对象实例
    km = KMeans(n_clusters=4)
    # 加载数据，进行训练，获得标签，总共是四个簇，就是四个标签，将给31个数据，每个数据都打上0-3的标签
    label = km.fit_predict(data)
    # 计算出每一个簇形成的所有的行内的数据，计算出该簇内的数据的和
    expenses = np.sum(km.cluster_centers_, axis=1)
    # 总共是四个标签，四个集合，按照打上的标签将城市名进行分类
    CityCluster = [[], [], [], []]
    # 遍历所有的标签，并将对应的城市根据标签加上对应的簇中
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    
    output_data = []
    # 将结果组织成字符串形式
    for i in range(len(CityCluster)):
        output_data.append("Expenses:%.2f\n" % expenses[i])
        output_data.append(str(CityCluster[i]) + '\n')
    # 打印列表
    for i in range(len(CityCluster)):
        print(f"Expenses:{expenses[i]:.2f}")
        print(CityCluster[i])

    return output_data

if __name__ == '__main__':
    input_file = r'.\sklearn\test_txt\city.txt'
    output_file = r'.\sklearn\test_txt\city_rst_4.txt'
    output_data = process_data(input_file)

    # 将输出结果保存到txt文件中
    with open(output_file, 'w', encoding='UTF-8') as f:
        f.writelines(output_data)

""" 注：前缀 r 表示一个原始字符串（raw string）。
使用原始字符串可以告诉Python解释器不对字符串中的反斜杠进行转义处理，而是将其视为普通字符
使用于Windows系统，其中的文件路径通常包含反斜杠"""