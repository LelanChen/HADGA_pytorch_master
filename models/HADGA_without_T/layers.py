# -*- coding: utf-8 -*-
# @Time    : 2024/5/6 10:32
# @Author  : chenlelan
# @File    : layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DefaultConfig

cfg = DefaultConfig()

class NeighborsAttentionLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim, n_heads,
                 bias=True, use_edge=True):
        super(NeighborsAttentionLayer, self).__init__()
        self.input_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = out_dim
        self.bias = bias
        self.n_heads = n_heads
        self.use_edge = use_edge
        self.n_calls = 0    # 邻居注意力层可以堆叠，从而对多跳邻居进行聚合，邻居注意力层数标识
        self.node_conv1d_layer = nn.ModuleList()
        self.edge_conv1d_layer = nn.ModuleList()
        self.x_dense_layer = nn.ModuleList()

        for j in range(self.n_heads):
            self.node_conv1d_layer.append(nn.Conv1d(in_channels=self.input_dim, out_channels=self.output_dim // self.n_heads, kernel_size=1))
            self.edge_conv1d_layer.append(nn.Conv1d(in_channels=self.edge_dim, out_channels=self.output_dim // self.n_heads, kernel_size=1))
            self.x_dense_layer.append(nn.Linear(self.input_dim, self.output_dim // self.n_heads))
        if use_edge:
            self.conv1 = nn.Conv1d(in_channels=(self.output_dim // self.n_heads) * 2, out_channels=self.output_dim // self.n_heads, kernel_size=1)
        else:
            self.conv1 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.output_dim // self.n_heads, kernel_size=1)
        self.dense2 = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, inputs):
        """
        inputs -- [[节点特征， 邻居节点特征， 邻居边缘特征]]
        use_edge -- 布尔值，表示是否使用边缘特征
        """
        self.n_calls += 1
        x = inputs[0]
        node_x = inputs[1]
        edge_x = inputs[2]
        attentions = []     # 邻居注意力层输出列表
        for j in range(self.n_heads):
            if self.use_edge:
                attentions.append(self.joint_attn_head(j, x, node_x=node_x, edge_x=edge_x,))

            else:
                attentions.append(self.attn_head(j, x, node_x=node_x))

        # 拼接多头注意力
        neighbors_h = torch.cat(attentions, dim=-1)
        # print('size of neighbors_h', neighbors_h.size())

        # 中心节点隐藏层
        # print('concat self_h and neighbor_h')
        self_h = self.dense2(x)     # [N, F]
        self_h = F.relu(self_h)

        # h = neighbors_h + self_h
        h = torch.cat([self_h, neighbors_h], dim=1)     # 拼接后输出的维度变为原来的两倍
        # print('concat self_h and neighbor_h size:', h.size())
        return h

    def joint_attn_head(self, j, x, node_x, edge_x):
        '''
        基于联合注意力聚合节点特征和边缘特征，捕获互补信息
        x -- 输入的中心节点特征
        node_x -- 采样的邻居节点特征
        edge_x -- 对应的采样边缘节点特征
        '''

        # 生成可学习参数Q和K-n，K-e
        # 注意pytorch中只能对倒数第2维数据进行卷积，因此传参时要转置一下，
        # 将需要卷积的数据弄到倒数第2维, 这里将embeding的维度进行卷积
        f_n = self.node_conv1d_layer[j](node_x.permute(0, 2, 1))    # [N, d, F]
        f_n = f_n.permute(0, 2, 1)
        # print("size of input node_x", node_x.size())
        # print("size of f_n", f_n.size())
        f_e = self.edge_conv1d_layer[j](edge_x.permute(0, 2, 1))    # [N, d, F]
        f_e = f_e.permute(0, 2, 1)
        # print("size of f_e", f_e.size())
        f_x = self.x_dense_layer[j](x)   # 通过fc实现权重变换[N,F]
        f_x = torch.unsqueeze(f_x, dim=1)   # [N,1,F]
        # print("size of f_x", f_x.size())

        # 参数 negative_slope 表示负值部分的斜率，默认为 0.01
        k_n = F.leaky_relu(f_n, negative_slope=0.02)      # [N, d, F]
        k_e = F.leaky_relu(f_e, negative_slope=0.02)
        q = F.leaky_relu(f_x, negative_slope=0.02)        # [N, 1, F]

        # 求邻居节点注意力权重
        logits_n = torch.matmul(q, k_n.transpose(1, 2))  # [N, 1, d]
        logits_n = logits_n / (k_n.size(-1) ** 0.5)   # 缩放
        coefficient_n = F.softmax(logits_n, dim=-1)  # [N, 1, d]  注意力权重
        # 求邻接边注意力权重
        logits_e = torch.matmul(q, k_e.transpose(1, 2))  # [N, 1, d]
        logits_e = logits_e / (k_e.size(-1) ** 0.5)  # 缩放
        coefficient_e = F.softmax(logits_e, dim=-1)
        # 联合注意力权重系数
        coefficients = F.softmax(coefficient_n + coefficient_e, dim=-1)     # 联合注意力权重系数

        cat_fts = torch.cat([f_n, f_e], dim=2)   # 将节点特征和边缘特征合并（可以选择相加/拼接）
        # print('shape of constant fets:', cat_fts.size())
        v = self.conv1(cat_fts.permute(0, 2, 1))
        v = F.leaky_relu(v.permute(0, 2, 1), negative_slope=0.02)  # [N, d, F]
        # print('shape of constant v:', v.size())

        # print('shape of joint attentions:', coefficients.shape)
        values = torch.matmul(coefficients, v)  # [N, 1, F]
        # print('shape of values:', values.shape)
        neighbor_h = torch.squeeze(F.relu(values))  # 默认删除所有为1的维度，[N, F]
        # print('shape of neighbor_h:', neighbor_h.shape)
        return neighbor_h

    def attn_head(self, j, x, node_x):
        f_n = self.node_conv1d_layer[j](node_x.permute(0, 2, 1))  # [N, d, F]
        f_n = f_n.permute(0, 2, 1)
        # print("size of input node_x", node_x.size())
        # print("size of f_n", f_n.size())
        f_x = self.x_dense_layer[j](x)  # 通过fc实现权重变换[N,F]
        f_x = torch.unsqueeze(f_x, dim=1)  # [N,1,F]
        # print("size of f_x", f_x.size())

        # 参数 negative_slope 表示负值部分的斜率，默认为 0.01
        k_n = F.leaky_relu(f_n, negative_slope=0.02)  # [N, d, F]
        q = F.leaky_relu(f_x, negative_slope=0.02)  # [N, 1, F]

        # 求邻居节点注意力权重
        logits_n = torch.matmul(q, k_n.transpose(1, 2))  # [N, 1, d]
        logits_n = logits_n / (k_n.size(-1) ** 0.5)  # 缩放
        coefficient_n = F.softmax(logits_n, dim=-1)  # [N, 1, d]  注意力权重

        v = self.conv1(node_x.permute(0, 2, 1))
        v = F.leaky_relu(v.permute(0, 2, 1), negative_slope=0.02)  # [N, d, F]
        # print('shape of constant v:', v.size())

        # print('shape of joint attentions:', coefficients.shape)
        values = torch.matmul(coefficient_n, v)  # [N, 1, F]
        # print('shape of values:', values.shape)
        neighbor_h = torch.squeeze(F.relu(values))  # 默认删除所有为1的维度，[N, F]
        # print('shape of neighbor_h:', neighbor_h.shape)
        return neighbor_h
