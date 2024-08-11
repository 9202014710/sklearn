"""
基于聚类的整图分割实例
"""

from ast import main
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans

def loadData(filePath):
        f = open(filePath,'rb')
        data = []
        img = image.open(f)
        m,n = img.size
        for i in range(m):
            for j in range(n):
                x, y, z = img.getpixel((i,j))  #getpixel
                data.append([x/256.0,y/256.0,z/256.0])
        f.close()
        #return np.mat(data), m, n
        return np.asarray(data), m, n  # 转换为 numpy 数组

def process_image(input_file):
    imgData, row, col = loadData(input_file)
    label = KMeans(n_clusters=8).fit_predict(imgData)
    label = label.reshape([row,col])
    print(label)
    pic_new = image.new("L", (row, col))
    for i in range(row):
        for j in range(col):
            pic_new.putpixel((i,j), int(256/(label[i][j]+1)))  
    output_file = input_file.replace('.jpg', '_2.jpg')
    pic_new.save(output_file, "JPEG")

# Example usage:
#input_file = r'.\sklearn\test_image\bull.jpg'
if __name__ == "__main__":
    input_file = r'.\sklearn\test_image\frog.jpg'
    process_image(input_file)
