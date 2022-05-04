import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn import metrics

def distance(vecA, vecB):
    '''
    计算两个向量之间欧氏距离的平方
    :param vecA: 向量A的坐标
    :param vecB: 向量B的坐标
    :return: 返回两个向量之间欧氏距离的平方
    '''
    dist = (vecA - vecB) * (vecA - vecB).T
    return dist[0, 0]


def randomCenter(data, k):
    '''
    随机初始化聚类中心
    :param data: 训练数据
    :param k: 聚类中心的个数
    :return: 返回初始化的聚类中心
    '''
    n = np.shape(data)[1]  # 特征的个数
    cent = np.mat(np.zeros((k, n)))  # 初始化K个聚类中心
    for j in range(n):  # 初始化聚类中心每一维的坐标
        minJ = np.min(data[:, j])
        rangeJ = np.max(data[:, j]) - minJ
        cent[:, j] = minJ * np.mat(np.ones((k, 1))) + np.random.rand(k, 1) * rangeJ  # 在最大值和最小值之间初始化
    return cent


def kmeans(data, k, cent):
    '''
    kmeans算法求解聚类中心
    :param data: 训练数据
    :param k: 聚类中心的个数
    :param cent: 随机初始化的聚类中心
    :return: 返回训练完成的聚类中心和每个样本所属的类别
    '''
    m, n = np.shape(data)  # m：样本的个数；n：特征的维度
    subCenter = np.mat(np.zeros((m, 2)))  # 初始化每个样本所属的类别
    change = True  # 判断是否需要重新计算聚类中心
    while change == True:
        change = False  # 重置
        for i in range(m):
            minDist = np.inf  # 设置样本与聚类中心的最小距离，初始值为正无穷
            minIndex = 0  # 所属的类别
            for j in range(k):
                # 计算i和每个聚类中心的距离
                dist = distance(data[i, ], cent[j, ])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            # 判断是否需要改变
            if subCenter[i, 0] != minIndex:  # 需要改变
                change = True
                subCenter[i, ] = np.mat([minIndex, minDist])
        # 重新计算聚类中心
        for j in range(k):
            sum_all = np.mat(np.zeros((1, n)))
            r = 0  # 每个类别中样本的个数
            for i in range(m):
                if subCenter[i, 0] == j:  # 计算第j个类别
                    sum_all += data[i, ]
                    r += 1
            for z in range(n):
                try:
                    cent[j, z] = sum_all[0, z] / r
                except:
                    print("ZeroDivisionError: division by zero")
    return subCenter, cent

def kmeanswithARI(latent_z, label_z):
    latent_z = np.squeeze(latent_z)
    labels_true = np.squeeze(label_z)
    k = 7  # 聚类中心的个数
    subCenter, center = kmeans(latent_z, k, randomCenter(latent_z, k))
    x, y, labels_pred = [], [], []
    for i in range(latent_z.shape[0]):
        point, predict = latent_z[i], subCenter.item(i, 0)
        point = latent_z[i]
        x.append(point[0])
        y.append(point[1])
        labels_pred.append(predict)
    # print(c1)
    # 绘制二维分类散点图矩阵
    fig1, ax1 = plt.subplots()
    scatter1 = ax1.scatter(x, y, marker='.', c=labels_true)
    legend1 = ax1.legend(*scatter1.legend_elements(), loc="lower left", title="Classes")
    ax1.add_artist(legend1)
    fig2, ax2 = plt.subplots()
    scatter2 = ax2.scatter(x, y, marker='.', c=labels_pred)
    legend2 = ax2.legend(*scatter2.legend_elements(), loc="lower left", title="Classes")
    ax2.add_artist(legend2)
    plt.show()
    score = metrics.adjusted_rand_score(labels_true, labels_pred)
    return score



