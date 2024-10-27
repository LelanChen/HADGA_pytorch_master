# -*- coding: utf-8 -*-
# @Time    : 2024/5/6 9:23
# @Author  : chenlelan
# @File    : tbatch.py

import numpy as np
import torch
from utils.shuffle_node import *
from utils.sample import *
from config import DefaultConfig

cfg = DefaultConfig()

class MinibatchIterator(object):
    """
    此小批量迭代器循环访问节点，以对一批节点的上下文对进行采样。

    graphs -- list of networkx graphs
    features -- list of node attribute matrices
    label -- list of node label matrices
    adjs -- list of adj matrices (of the graphs)
    edge_features -- list of edge attribute matrices
    num_features -- dimension of node attribute
    num_edge_features -- dimension of edge attribute
    """
    def __init__(self, graphs, adjs, label, classes, features, edge_features, num_features, num_edge_features):

        self.graphs = graphs
        self.adjs = adjs
        self.label = label
        self.classes = classes
        self.features = features    # 节点属性特征
        self.edge_features = edge_features
        self.num_features = num_features    # 节点属性的维度
        self.num_edge_features = num_edge_features  # 边缘属性的维度
        self.start_time_step = 0   # 每个批次的开始时间
        self.num_time_steps = len(graphs)
        self.node_ids = list([list(self.graphs[t].nodes()) for t in range(0, self.num_time_steps)]) # all nodes in the graphs.

    def end(self):
        return self.start_time_step >= self.num_time_steps

    def batch_feed_dict(self, batch_nodes):
        """
        feed -- 字典['node_ids', 'features', 'neighbor_node_features', 'neighbor_edge_features', 'label']
        包含 （a） 节点id、（b） 属性矩阵列表、（c） 采样邻居节点特征、（d）采样邻居边缘特征、（e）标签的 feed dict"""
        min_t = max(0, self.start_time_step - cfg.window)
        # print('min_t', min_t)
        max_t = self.start_time_step - 1
        # print('max_t', max_t)
        feed_dict = dict()
        assert len(batch_nodes) == len(self.label[self.start_time_step - 1])
        
        remain_node, remain_label, remain_feat, remain_index = self.filter(batch_nodes)
        # remain_node_ids_all = np.zeros((len(remain_node), cfg.window))  # 初始化剩余节点列表[N, T]
        remain_feat_all = np.zeros((len(remain_node), cfg.window, self.num_features))  # 初始化节点特征列表[N, T, S]
        sample_node_feat_all = np.zeros(
            (len(remain_node), cfg.window, cfg.neighbor_sample_size, self.num_features))  # 初始化采样节点特征列表[N, T, d,S]
        sample_edge_feat_all = np.zeros(
            (len(remain_node), cfg.window, cfg.neighbor_sample_size, self.num_edge_features))  # 初始化采样边缘特征列表[N, T, d,E]

        index = 0
        for node in remain_node:
            for t in range(min_t, max_t + 1):
                if node in self.node_ids[t]:
                    # remain_node_ids_all[index][t - min_t] = node  # [N, T]
                    remain_feat_all[index][t - min_t] = self.features[t][node]  # [N, T, S]
                    # 进行邻居采样
                    node_index = list(self.node_ids[t]).index(node)  # 节点在第t个快照中的索引
                    neighbor_index = one_sampling(node_index, self.adjs[t], cfg.neighbor_sample_size)  # [d,]
                    assert len(neighbor_index) == cfg.neighbor_sample_size, '采样节点数与cfg.neighbor_sample_size不匹配'
                    # print("采样索引", neighbor_index)

                    sample_node_fts = []
                    sample_edge_fts = []
                    for idx in neighbor_index:
                        if idx == 'null':
                            # 出度为0，没有采样邻居，用0填充采样节点特征和边缘特征，使输入大小一致
                            sample_node_fts.append(np.zeros(self.num_features))
                            sample_edge_fts.append(np.zeros(self.num_edge_features))
                        else:
                            # self.node_features是由T个时刻的dict组成的列表， [dict{node_id： node_feature},...]
                            sample_node_fts.append(self.features[t][list(self.node_ids[t])[idx]])  # [d, S]
                            # self.edge_features是由T个时刻的dict组成的列表， [dict{(node_id, neighbor_id)： edge_feature},...]
                            sample_edge_fts.append(self.edge_features[t][(node, list(self.node_ids[t])[idx])])  # [d, E]
                    sample_node_feat_all[index][t - min_t] = sample_node_fts  # [N, T, d, S]
                    sample_edge_feat_all[index][t - min_t] = sample_edge_fts  # [N, T, d, E]
            index = index + 1

        feed_dict.update({'node_ids': torch.IntTensor(remain_node)})
        feed_dict.update({'remain_index': remain_index})
        feed_dict.update({'label': torch.IntTensor(remain_label)})
        feed_dict.update({'features': torch.from_numpy(remain_feat_all).float()})
        feed_dict.update({'ngh_node_features': torch.from_numpy(sample_node_feat_all).float()})
        feed_dict.update({'ngh_edge_features': torch.from_numpy(sample_edge_feat_all).float()})
        feed_dict.update({'batch_node': torch.IntTensor(batch_nodes)})
        feed_dict.update({'batch_label': torch.IntTensor(self.label[self.start_time_step - 1])})

        return feed_dict

    def next_batch_generator(self):
        """
        生成器函数，用于生成各个时刻快照的小批次。一个快照的节点当作一个batch.
        self.node_ids (list): 所有时刻的节点ID列表，维度为[T, N_t]，T表示时间快照数量，N_t表示每个时刻的节点数量（N_t不一致）。
        返回:
        batche (generator): 生成器，每次生成一个小批次
        feed_dict:
        """
        batch = []
        # 填充当前批次
        batch.extend(self.node_ids[self.start_time_step])
        # print(self.start_time_step)
        # print(len(batch))
        self.start_time_step = self.start_time_step + 1
        return self.batch_feed_dict(batch)

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch start time.
        """
        self.node_ids,self.label,self.adjs,self.classes = shuffle_node(self.node_ids, self.label, self.adjs, self.classes)
        self.start_time_step = 0

    def filter(self, batch_nodes):
        '''
        过滤出度为0的节点，即没有发送数据包的节点，这类节点一般为正常节点，以提高计算效率
        '''
        remain_node = []
        remain_label = []
        remain_features = []
        remain_index = []
        # print("self.start_time_step", self.start_time_step)
        for i in range(len(batch_nodes)):
            # 获取第i个节点的所有邻居节点索引
            neighbors_idx = np.nonzero(self.adjs[self.start_time_step - 1][i])[0]
            if len(neighbors_idx) > 0:
                # 保留出度大于0的节点
                remain_index.append(i)
                remain_node.append(list(batch_nodes)[i])
                remain_label.append(self.label[self.start_time_step - 1][i])
                remain_features.append(self.features[self.start_time_step - 1][list(batch_nodes)[i]])
            else:
                pass
        # print('remain_node', remain_node)
        # print('remain_label', remain_label)
        # print('remain_feat', remain_features)
        return remain_node, remain_label, remain_features, remain_index