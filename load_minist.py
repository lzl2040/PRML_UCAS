# -*- encoding: utf-8 -*-
"""
File load_minist.py
Created on 2024/1/20 18:55
Copyright (c) 2024/1/20
@author: 
"""
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

def load_minist_data(minist_path):
    # 加载mat格式数据
    mnist = loadmat(minist_path)
    # N
    minist_data_labels = mnist["label"][0]
    print(f"data labels shape:{minist_data_labels.shape}")
    # N D
    minist_data = mnist['data'].T
    print(f"data shape:{minist_data.shape}")
    return minist_data, minist_data_labels


if __name__ == '__main__':
    minist_path = "E:\\Desktop\\在此学习\\研究生\\课程学习\\模式识别\\实践作业\\mnist-original.mat"
    data, labels = load_minist_data(minist_path)
    # 可视化数据
    data_1 = data[1].reshape(28, 28)
    # data_1[data_1 != 0] = 1
    plt.imshow(data_1)
    plt.show()
    # 数据里面的唯一value
    print(f"unique values:{np.unique(data_1)}")
