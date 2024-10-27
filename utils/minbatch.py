# -*- coding: utf-8 -*-
# @Time    : 2024/5/10 8:58
# @Author  : chenlelan
# @File    : minbatch.py

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

    def __init__(self, graphs, adjs, label, classes, features, edge_features, num_features, num_edge_features, batch_size = cfg.batch_size):

        self.graphs = graphs
        self.adjs = adjs
        self.label = label  # 二分类标签
        self.classes = classes  # 多分类标签
        self.features = features  # 节点属性特征
        self.edge_features = edge_features
        self.num_features = num_features  # 节点属性的维度
        self.num_edge_features = num_edge_features  # 边缘属性的维度
        self.start_time_step = 0  # 每个批次的开始时间
        self.batch_size = batch_size
        self.num_time_steps = len(graphs)
        self.node_ids = list(
            [list(self.graphs[t].nodes()) for t in range(0, self.num_time_steps)])  # all nodes in the graphs.

    def end(self):
        return self.start_time_step >= self.num_time_steps

    def batch_feed_dict(self, batch_nodes):
        """
        feed -- 字典['node_ids', 'features', 'neighbor_node_features', 'neighbor_edge_features', 'label']
        包含 （a） 节点id、（b） 属性矩阵列表、（c） 采样邻居节点特征、（d）采样邻居边缘特征、（e）标签的 feed dict"""
        time_step = len(batch_nodes)
        remain_node_all = []
        remain_label_all = []
        remain_classes_all = []
        remain_index_all = []
        remain_feat_all = []
        sample_node_feat_all = []
        sample_edge_feat_all = []
        index_offset = 0   # 索引偏移量
        for i in range(time_step):
            min_t = max(0, self.start_time_step - time_step + i - cfg.window + 1)   # min_t = max_t - window + 1
            # print('min_t', min_t)
            max_t = self.start_time_step - time_step + i    # max_t其实就是当前快照所在的时隙
            # print('max_t', max_t)
            feed_dict = dict()
            assert len(batch_nodes[i]) == len(self.label[self.start_time_step - time_step + i])

            remain_node, remain_label, remain_classes, remain_index = self.filter(batch_nodes[i], max_t, index_offset)
            index_offset = index_offset + len(batch_nodes[i])
            remain_node_all.extend(remain_node)
            remain_label_all.extend(remain_label)
            remain_classes_all.extend(remain_classes)
            # print("remain_classes_all", remain_classes_all)
            remain_index_all.extend(remain_index)
            # remain_node_ids_all = np.zeros((len(remain_node), cfg.window))  # 初始化剩余节点列表[N, T]
            remain_feat = np.zeros((len(remain_node), cfg.window, self.num_features))  # 初始化节点特征列表[N, T, S]
            sample_node_feat = np.zeros(
                (len(remain_node), cfg.window, cfg.neighbor_sample_size, self.num_features))  # 初始化采样节点特征列表[N, T, d,S]
            sample_edge_feat = np.zeros(
                (len(remain_node), cfg.window, cfg.neighbor_sample_size, self.num_edge_features))  # 初始化采样边缘特征列表[N, T, d,E]

            index = 0
            for node in remain_node:
                for t in range(min_t, max_t + 1):
                    if node in self.node_ids[t]:
                        # remain_node_ids_all[index][t - min_t] = node  # [N, T]
                        remain_feat[index][t - min_t] = self.features[t][node]  # [N, T, S]
                        # 进行邻居采样
                        node_index = list(self.node_ids[t]).index(node)  # 节点在第t个快照中的索引
                        neighbor_index = one_sampling(node_index, self.adjs[t], cfg.neighbor_sample_size)  # [d,]
                        assert len(neighbor_index) == cfg.neighbor_sample_size, '采样节点数与cfg.neighbor_sample_size不匹配'
                        # print("采样索引", neighbor_index)

                        node_fts = []
                        edge_fts = []
                        for idx in neighbor_index:
                            if idx == 'null':
                                # 出度为0，没有采样邻居，用0填充采样节点特征和边缘特征，使输入大小一致
                                node_fts.append(np.zeros(self.num_features))
                                edge_fts.append(np.zeros(self.num_edge_features))
                            else:
                                # self.node_features是由T个时刻的dict组成的列表， [dict{node_id： node_feature},...]
                                node_fts.append(self.features[t][list(self.node_ids[t])[idx]])  # [d, S]
                                # self.edge_features是由T个时刻的dict组成的列表， [dict{(node_id, neighbor_id)： edge_feature},...]
                                edge_fts.append(self.edge_features[t][(node, list(self.node_ids[t])[idx])])  # [d, E]
                        sample_node_feat[index][t - min_t] = node_fts  # [N_t, T, d, S]
                        sample_edge_feat[index][t - min_t] = edge_fts  # [N_t, T, d, E]
                index = index + 1
            remain_feat_all.extend(remain_feat)     # [batch_size, T, S]
            sample_node_feat_all.extend(sample_node_feat)    # [batch_size, T, d, S]
            sample_edge_feat_all.extend(sample_edge_feat)    # [batch_size, T, d, S]

        feed_dict.update({'node_ids': torch.IntTensor(remain_node_all)})
        feed_dict.update({'remain_index': remain_index_all})
        feed_dict.update({'label': torch.IntTensor(remain_label_all)})
        feed_dict.update({'classes': torch.IntTensor(np.array(remain_classes_all))})
        feed_dict.update({'features': torch.from_numpy(np.array(remain_feat_all)).float()})
        feed_dict.update({'ngh_node_features': torch.from_numpy(np.array(sample_node_feat_all)).float()})
        feed_dict.update({'ngh_edge_features': torch.from_numpy(np.array(sample_edge_feat_all)).float()})
        batch_nodes_all = list(sum(batch_nodes, []))    # 将多个子列表拼接在一起
        feed_dict.update({'batch_node': torch.IntTensor(batch_nodes_all)})
        batch_label_all = list(sum(self.label[(self.start_time_step - time_step): self.start_time_step], []))
        feed_dict.update({'batch_label': torch.IntTensor(batch_label_all)})
        batch_classes_all = list(sum(self.classes[(self.start_time_step - time_step): self.start_time_step], []))
        feed_dict.update({'batch_classes': torch.IntTensor(np.array(batch_classes_all))})

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
        end_time = self.start_time_step
        count = 0
        # 填充当前批次
        while(end_time < self.num_time_steps and count < self.batch_size):
            batch.append(self.node_ids[end_time])
            count = count + len(self.node_ids[end_time])
            end_time = end_time + 1

        self.start_time_step = end_time
        # print(self.start_time_step)
        # print(len(batch))
        return self.batch_feed_dict(batch)

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch start time.
        """
        self.node_ids, self.label, self.adjs, self.classes = shuffle_node(self.node_ids, self.label, self.adjs, self.classes)
        self.start_time_step = 0

    def filter(self, batch_nodes, t, index_offset):
        '''
        过滤出度为0的节点，即没有发送数据包的节点，这类节点一般为正常节点，以提高计算效率
        batch_nodes -- 一个批次中第 i 个时隙的节点列表
        t -- 这一快照所在的时隙
        index_offset -- 最小批次内前几个时隙节点索引的偏移量
        '''
        remain_node = []
        remain_label = []
        remain_classes = []
        remain_index = []
        # print("self.start_time_step", self.start_time_step)
        for i in range(len(batch_nodes)):
            # 获取第i个节点的所有邻居节点索引
            neighbors_idx = np.nonzero(self.adjs[t][i])[0]
            if len(neighbors_idx) > 0:
                # 保留出度大于0的节点
                remain_index.append(i + index_offset)
                remain_node.append(list(batch_nodes)[i])
                remain_label.append(self.label[t][i])
                remain_classes.append(self.classes[t][i])
            else:
                pass
        # print('remain_node', remain_node)
        # print('remain_label', remain_label)
        # print('remain_index', remain_index)
        # print('remain_classes', remain_classes)
        return remain_node, remain_label, remain_classes, remain_index
