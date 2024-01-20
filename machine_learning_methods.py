# -*- encoding: utf-8 -*-
"""
File machine_learning_methods.py
Created on 2024/1/20 18:55
Copyright (c) 2024/1/20
@author: 
"""
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from load_minist import load_minist_data

if __name__ == '__main__':
    minist_path = "E:\\Desktop\\在此学习\\研究生\\课程学习\\模式识别\\实践作业\\mnist-original.mat"
    method_type = "linear_svm"
    X_data, Y_data = load_minist_data(minist_path)
    # 数据规范化
    scaler = StandardScaler()
    X = scaler.fit_transform(X_data)
    # 分割得到训练和测试数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=10000, random_state=42)
    print(f"Train data size:{X_train.shape}")
    print(f"Test data size:{X_test.shape}")
    if method_type == "linear_svm":
        print("Start training Linear SVM...")
        # 构建linear svm C表示正则项的权重
        l_svm = svm.LinearSVC(C = 10, max_iter=2000)
        l_svm.fit(X_train, Y_train)
        print("Training over!")
        print("The function is:")
        print(f"w:{l_svm.coef_}")
        print(f"b:{l_svm.intercept_}")

        print("Start testing...")
        # 打印模型的精确度
        print(f"{l_svm.score(X_test, Y_test) * 100}%")
    elif method_type == "kernel_svm":
        print("Start training Kernel SVM...")
        # 构建linear svm C表示正则项的权重
        k_svm = svm.SVC(C=100, max_iter=1000)
        k_svm.fit(X_train, Y_train)
        print("Training over!")

        print("Start testing...")
        # 打印模型的精确度
        print(f"{k_svm.score(X_test, Y_test) * 100}%")
    elif method_type == "decision_tree":
        print("Start training Decision Tree...")
        # 构建决策树
        d_tree = DecisionTreeClassifier(criterion = "gini", splitter = "best")
        d_tree.fit(X_train, Y_train)
        print("Training over!")

        print("Start testing...")
        # 打印模型的精确度
        print(f"{d_tree.score(X_test, Y_test) * 100}%")