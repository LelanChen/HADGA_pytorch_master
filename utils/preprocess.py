# -*- encoding:utf-8 -*-
from __future__ import print_function

import os
import torch
import numpy as np
import networkx as nx
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler


def load_graphs(dataset_str, file):
    """加载给定数据集中的图形快照"""
    graphs = np.load("data/{}/{}".format(dataset_str, file), allow_pickle=True)['graph']
    print("Loaded {} graphs ".format(len(graphs)))
    adj_matrices = list(map(lambda x: np.array(nx.adjacency_matrix(x).todense()), graphs))
    labels = list(map(lambda g: [data.get('label') for _, data in g.nodes(data=True)], graphs))
    classes = list(map(lambda g: [data.get('classes') for _, data in g.nodes(data=True)], graphs))
    return graphs, adj_matrices, labels, classes

def get_label(graph):
    label = []
    for n, data in graph.nodes(data=True):
        l = data.get('label')
        label.append(l)
    return label


def load_feats(dataset_str, node_file, edge_file):
    """ 加载快照节点属性和边缘属性"""
    n_features = np.load("data/{}/{}".format(dataset_str, node_file), allow_pickle=True)['node_feat']
    print("Loaded {} X_n matrices ".format(len(n_features)))
    e_features = np.load("data/{}/{}".format(dataset_str, edge_file), allow_pickle=True)['edge_feat']

    return n_features, e_features


def sparse_to_tuple(sparse_mx):
    """将 scipy 稀疏矩阵转换为元组表示(for tf feed dict)."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):    # 检测mx是否是COO(三元组(row, col, data))的稀疏矩阵格式
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()    # vstack()垂直（按照行顺序）的把数组给堆叠起来
        values = mx.data
        shape = mx.shape
        return coords, values, shape

def features_standard(feat_df, columns):
    '''
    在构建通信图阶段就对特征数据进行最大最小归一化处理，防止在构建图快照之后，特征被划分成[T,N_t,F]维度时不方便进行归一化
    输入
    feat_df: 为dataframe格式，
    columns: 为需要归一化的列索引，去除时间戳和节点IP以及标签,仅保留特征所在列索引
    '''
    # 初始化 MinMaxScaler
    scaler = MinMaxScaler()
    # 对指定列进行最大最小规范化
    feat_df[columns] = scaler.fit_transform(feat_df[columns])

    # 打印规范化后的 DataFrame
    # print("Normalized DataFrame:")
    # print(feat_df)
    return feat_df

def one_hot(df):
    # df = pd.read_csv(data_path)
    labels = df["Attack"]
    unique_labels = np.unique(labels)
    one_hot_matrix = np.zeros((len(labels), len(unique_labels)))
    for i, label in enumerate(labels):
        one_hot_matrix[i, np.where(unique_labels == label)[0][0]] = 1
    one_hot_matrix = one_hot_matrix.astype(int)
    print(one_hot_matrix)
    # one_hot_df = pd.DataFrame(one_hot_matrix.astype(int), columns=unique_labels)
    # 将 one-hot 编码转换为张量
    # one_hot_tensor = torch.tensor(one_hot_df.values)
    # print(one_hot_tensor)
    # df["Class"] = one_hot_df.apply(lambda x: "".join(str(v) for v in x), axis=1)
    # df.to_csv(save_path, index=False)
    return one_hot_matrix



