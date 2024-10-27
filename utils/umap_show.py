# -*- coding: utf-8 -*-
# @Time    : 2024/5/30 9:06
# @Author  : chenlelan
# @File    : umap_show.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import umap
import seaborn as sns


def umap_show(data_df):
    # feature = data_df.iloc[:, 2: -2].head(5000).values
    # label = data_df.iloc[:, -2].head(5000).values

    feature = data_df.iloc[:, 0: -1].head(5000).values
    label = data_df.iloc[:, -1].head(5000).values

    labels = []
    for i in range(len(label)):
        if "1" in label[i]:
        # if label[i] == 1:
            labels.append("Attack")
        else:
            labels.append("Benign")

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(feature)

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    # colors = ['#17becf', '#d62728']
    # label_names = ['Benign', 'Bot']
    # point = [(14, 19.5), (14, 18)]
    # 绘制二维点图
    # plt.figure(figsize=(5, 5))
    # for i in range(len(colors)):
    #     mask = label == i
    #     plt.scatter(embedding[mask, 0], embedding[mask, 1], c=colors[i], s=10)
    #     plt.scatter(point[i][0], point[i][1], c=colors[i])  # 绘制标记点
    #     plt.annotate(label_names[i], (point[i][0], point[i][1]), textcoords="offset points", xytext=(5, -3))
    # 添加右上角的颜色类标签名称
    # plt.text(0.85, 0.95, 'Benign', transform=plt.gca().transAxes, color='#17becf')
    # plt.text(0.85, 0.9, 'Bot', transform=plt.gca().transAxes, color='#d62728')
    # plt.xlim(-20, 22)
    # plt.ylim(-20, 22)
    # plt.xlabel('UMAP 1')
    # plt.ylabel('UMAP 2')

    dftsne = pd.DataFrame(data=embedding, columns=['UMAP 1', 'UMAP 2'])
    dftsne['cluster'] = labels
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=dftsne, x='UMAP 1', y='UMAP 2', hue='cluster', legend="brief", palette='tab10', alpha=0.8)
    # 设置 x 轴和 y 轴标签的字体和大小
    # plt.xlabel('(a)raw traffic features', fontsize=12)
    plt.xlabel('(b)node embeddings', fontsize=12)

    # 设置图例的字体和大小
    plt.legend(fontsize=10)

    plt.show()

# raw data
# data_path = "D:/科研/小论文/数据集/Bot_IoT/features-5/test-1500/node_feature.csv"
data_path = "D:/科研/小论文/HADGA-pytorch/logs/HADGA/output/node-embedding.csv"
data_df = pd.read_csv(data_path)
umap_show(data_df)