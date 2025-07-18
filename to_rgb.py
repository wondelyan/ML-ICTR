 # -*- coding: utf-8 -*-
import imageio
import numpy as np
import pickle

# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

dataName = "data/cifar-10-batches-py/data_batch_1"  # 读取当前目录下的data_batch12345文件，dataName其实也是data_batch文件的路径，本文和脚本文件在同一目录下。
Xtr = unpickle(dataName)
print(dataName + " is loading...")

for i in range(0, 10):
    img = np.reshape(Xtr['data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
    img = img.transpose(1, 2, 0)  # 读取image
    picName = 'train/' + str(Xtr['labels'][i]) + '_' + str(i) + '.jpg'  # Xtr['labels']为图片的标签，值范围0-9，本文中，train文件夹需要存在，并与脚本文件在同一目录下。
    imageio.imsave(picName, img)
print(dataName + " loaded.")

