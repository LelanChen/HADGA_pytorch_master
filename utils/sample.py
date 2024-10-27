# -*- coding: utf-8 -*-
# @Time    : 2024/3/18 21:46
# @Author  : chenlelan
# @File    : sample.py
import numpy as np

def sample_neighbors(node_ids, adj, d):
    """
    对每个时刻的图快照中的所有点进行邻居节点采样，采样数为d，有放回
    参数:
    node_ids (list): 每个时刻的节点id列表，长度为T
    adj (list): 每个时刻的邻接矩阵列表，长度为T，每个邻接矩阵为形状为(N_t, N_t)的numpy数组
    d (int): 采样数量
    返回:
    sampled_ngh (list): 采样的邻居节点索引，shape为(N_t, d)
    """
    # 从邻接矩阵中每个节点的邻居中进行有放回的采样
    sampled_ngh = []
    for i in range(len(node_ids)):
        # 获取第i个节点的所有邻居节点索引
        neighbors_idx = np.nonzero(adj[i])[0]
        # 如果邻居数量小于采样数量，则随机重复采样直至达到采样数量
        if len(neighbors_idx) == 0:
            # 出度为0的节点，即没有发送数据包的节点，这类节点一般为正常节点，过滤掉此类节点可以提高计算效率
            pass
        elif len(neighbors_idx) < d:
            sampled_ngh.append(np.random.choice(neighbors_idx, size=d, replace=True))
        else:
            sampled_ngh.append(np.random.choice(neighbors_idx, size=d, replace=False))

    return sampled_ngh


def one_sampling(node_index, adj, d):
    """
    对节点进行一阶邻居节点采样，采样数为d，有放回
    参数:
    node_index : 中心节点的索引，用于查找对应行的邻接矩阵
    adj : 当前时刻的邻接矩阵列表，形状为(N_t, N_t)的numpy数组
    d (int): 采样数量
    返回:
    sampled_ngh (list): 采样的邻居节点索引，shape为(d,)
    """
    # 获取当前节点的所有邻居节点索引
    neighbors_idx = np.nonzero(adj[node_index])[0]
    sampled_ngh = ['null'] * d
    # print("初始化采样邻居索引", sampled_ngh)
    if len(neighbors_idx) == 0:
        # 出度为0的节点，即没有发送数据包的节点，这类节点一般为正常节点，过滤掉此类节点可以提高计算效率
        pass
    elif len(neighbors_idx) < d:
        # 如果邻居数量小于采样数量，则随机重复采样直至达到采样数量
        # sampled_ngh = np.random.choice(neighbors_idx, size=d, replace=True)
        sampled_ngh[: len(neighbors_idx)] = neighbors_idx
    else:
        # 随机无回放采样
        sampled_ngh = np.random.choice(neighbors_idx, size=d, replace=False)

    return sampled_ngh


def multihop_sampling(src_node, adj, sample_nums):
    '''
    多跳采样
    :param src_node: 源节点
    :param adj: 邻接矩阵
    :param sample_nums: 每阶需要采样的节点数
    :return: [源节点， 一跳邻居， 两跳邻居...]
    '''
    sampling_result = [src_node]
    for k, hopk_num in enumerate(sample_nums):
        if sampling_result[k] != 'null':
            hopk_result = one_sampling(sampling_result[k], adj, hopk_num)
        else:
            hopk_result = ['null'] * hopk_num
        sampling_result.append(hopk_result)
    return sampling_result
