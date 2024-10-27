# -*- coding: utf-8 -*-
# @Time    : 2024/4/8 18:49
# @Author  : chenlelan
# @File    : shuffle_node.py

import random
import numpy as np

def shuffle_node(nodes_list, labels_list, adj_list, classes_list):
    '''
    input：nodes_list 是一个包含了不规则的 2 维节点 id 的列表，维度为 [T, N_t]，
    labels_list 是与之对应的标签列表,
    classes_list 是与之对应的标签列表，
    feats_list是与之对应的节点特征
    adj是与之对应的邻接矩阵
    '''
    # 生成包含节点 id 和原始标签的元组列表
    origianl_index = []
    for t in range(len(nodes_list)):
        node_index = {}
        for i in range(len(nodes_list[t])):
            node_id = nodes_list[t][i]
            node_index[node_id] = i
        origianl_index.append(node_index)
    # print('origianl_index', origianl_index)

    shuffle_label_list = []
    shuffle_classes_list = []
    # shuffle_feats_list = []
    shuffle_adj_list = []
    # 对每个时刻的node列表进行 shuffle 处理，打乱节点顺序
    for t in range(len(nodes_list)):
        shuffle_label = []
        shuffle_classes = []
        shuffle_feats = []
        shuffle_adj = np.zeros_like(adj_list[t])
        random.shuffle(nodes_list[t])  # 对node列表进行 shuffle
        for i in range(len(nodes_list[t])):
            index = origianl_index[t][nodes_list[t][i]]
            shuffle_label.append(labels_list[t][index])
            shuffle_classes.append(classes_list[t][index])
            # shuffle_feats.append(feats_list[t][index])
            shuffle_adj[i] = adj_list[t][index]    # 根据原始顺序中的索引，将对应行复制到打乱后的邻接矩阵中
        # 根据打乱后的节点顺序重新排列adj列
        shuffle_adj = shuffle_adj[:, np.array([origianl_index[t][node] for node in nodes_list[t]])]
        shuffle_label_list.append(shuffle_label)
        shuffle_classes_list.append(shuffle_classes)
        # shuffle_feats_list.append(shuffle_feats)
        # print('shuffle_adj', shuffle_adj)
        shuffle_adj_list.append(shuffle_adj)

    return nodes_list, shuffle_label_list, shuffle_adj_list, shuffle_classes_list

'''# 示例
node_list = [[1,2,3,4],
             [2, 3, 6],
             [1, 3, 4, 6]]
label_list = [[1, 1, 0, 0],
              [0, 0 ,1],
              [0, 1, 0 ,1]]
feat_list = [[[1,1,1,1], [2,2,2,2], [3,3,3,3], [4,4,4,4]],
             [[2,2,2,2], [3,3,3,3], [6,6,6,6]],
             [[1,1,1,1], [3,3,3,3], [4,4,4,4], [6,6,6,6]]]
adj_list = [[[0,1,0,0],[1,0,0,1],[0,0,0,1],[0,1,1,0]],
            [[0,0,1],[0,1,1],[1,1,0]],
            [[0,1,0,0],[1,0,0,1],[0,0,0,1],[0,1,1,0]]]
shuffle_node, shuffle_label, shuffle_feats, shuffle_adj = shuffle_node(node_list, label_list, feat_list, adj_list)
print('shuffle_node', shuffle_node)
print('shuffle_label', shuffle_label)
print('shuffle_feats', shuffle_feats)
print('shuffle_adj', shuffle_adj)'''