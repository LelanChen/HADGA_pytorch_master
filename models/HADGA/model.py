# -*- coding: utf-8 -*-
# @Time    : 2024/5/6 15:28
# @Author  : chenlelan
# @File    : model.py

from models.HADGA.layers import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Recall, Precision, AUC
from sklearn.metrics import roc_curve, auc
from config import DefaultConfig

cfg = DefaultConfig()

class HADGA(nn.Module):
    def __init__(self, node_dim, edge_dim, use_edge):
        super(HADGA, self).__init__()
        self.neighbors_attention_layers = nn.ModuleList()
        self.temporal_attention_layers = nn.ModuleList()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.use_edge = use_edge  # 是否使用边缘信息
        self.neighbors_head_config = cfg.neighbors_head_config
        self.neighbors_layer_config = cfg.neighbors_layer_config
        self.temporal_head_config = cfg.temporal_head_config
        self.temporal_layer_config = cfg.temporal_layer_config
        # self.spatial_drop = spatial_drop
        # self.temporal_drop = temporal_drop
        self._build()
        self.loss = 0

    def _build(self):
        input_dim = self.node_dim
        edge_dim = self.edge_dim

        # 1: Neighbors Attention Layers
        for i in range(0, len(self.neighbors_layer_config)):
            if i > 0:
                input_dim = self.neighbors_layer_config[i - 1]
            self.neighbors_attention_layers.append(NeighborsAttentionLayer(node_dim=input_dim, edge_dim=edge_dim,
                                                                           out_dim=self.neighbors_layer_config[i],
                                                                           n_heads=self.neighbors_head_config[i],
                                                                           use_edge=self.use_edge))

        # 2: Temporal Attention Layers
        input_dim = self.neighbors_layer_config[-1] * 2
        for i in range(0, len(self.temporal_layer_config)):
            if i > 0:
                input_dim = self.temporal_layer_config[i - 1]
            out_dim = self.temporal_layer_config[i]
            self.temporal_attention_layers.append(TemporalAttentionLayer(input_dim=input_dim, out_dim=out_dim,
                                                                         n_heads=self.temporal_head_config[i],
                                                                         max_time_steps=cfg.window,
                                                                         residual=False))

        # 3: 创建全连接层，实现二元分类
        self.embed_layer = nn.Linear(self.temporal_layer_config[-1], 8)
        # self.fc_layer = nn.Linear(self.temporal_layer_config[-1], 1)
        self.fc_layer = nn.Linear(8, 1)

    def forward(self, placeholders, training=False):
        # 1: Neighbors Attention forward
        self.placeholders = placeholders
        self.training = training
        input_tensor = placeholders['features']  # List of centre node feature matrices. [N_t, T, S]
        neighbors_node_fts = placeholders['ngh_node_features']  # [N_t, T, d, S]
        neighbors_edge_fts = placeholders['ngh_edge_features']  # [N_t, T, d, E]
        spatial_drop = placeholders['spatial_drop']
        N = input_tensor.shape[0]
        print('处理的节点数量为', N)
        # 将T个快照特征按行拼接成[T*N_t, S]的规则张量,设M = N_t * T
        flat_input = input_tensor.view(-1, self.node_dim)  # [M, S]
        flat_node_fts = neighbors_node_fts.view(-1, cfg.neighbor_sample_size, self.node_dim)  # [M, d, S]
        flat_edge_fts = neighbors_edge_fts.view(-1, cfg.neighbor_sample_size, self.edge_dim)  # [M, d, E]
        # print('shape of flat_input', flat_input.size(), type(flat_input))
        for idx, layer in enumerate(self.neighbors_attention_layers):
            self.neighbor_output_embeddings = layer([flat_input, flat_node_fts, flat_edge_fts])  # [M, F]
            self.neighbor_output_embeddings = F.dropout(self.neighbor_output_embeddings, p=spatial_drop, training=self.training)
            # 如果要聚合多跳邻居节点特征，需要进行多阶采样

        print('neighbor model final_output_embeddings shape', self.neighbor_output_embeddings.size())
        # for v in self.trainable_variables:
        #     if ':' not in v.name:
        #         print('unreadvariable')
        #         pdb.set_trace()

        # 2：还原邻居注意力层的输出形状为[N, T, F],作为时间注意力层的输入
        self.temporal_input = self.neighbor_output_embeddings.view(-1, input_tensor.shape[1],
                                                (self.neighbors_layer_config[-1] * 2))  # [N, T, 2*F]
        print('temporal model input shape', self.temporal_input.size())


        # 3: Temporal Attention forward
        temporal_drop = self.placeholders['temporal_drop']
        for idx, temporal_layer in enumerate(self.temporal_attention_layers):
            outputs = temporal_layer(self.temporal_input)  # [N, T, F]
            outputs = F.dropout(outputs, p=temporal_drop, training=self.training)
            self.temporal_input = outputs

        # 使用索引操作获取最后一个时间步的输出，并对结果进行压缩
        self.final_output_embeddings = outputs[:, -1, :].squeeze()  # [N, F]
        print('shape of final_output_embeddings', self.final_output_embeddings.size())
        # self.logist = self.fc_layer(self.final_output_embeddings)  # [N, 1]
        embed = self.embed_layer(self.final_output_embeddings)
        self.logist = self.fc_layer(embed)
        logist = torch.sigmoid(self.logist)
        self.labels = placeholders['label']
        self.pred_y = torch.squeeze(torch.round(logist).to(torch.int32))  # (四舍五入取整）预测标签

        remain_index = self.placeholders['remain_index']  # list
        batch_label = self.placeholders['batch_label']  # [N_t]tensor
        pred_y = np.zeros(batch_label.shape[0])  # 初始化预测标签, [batchsize]array
        # print('pred_y', self.pred_y)
        # print('number of remain_index', remain_index)
        # print('batch_label', batch_label)
        assert self.pred_y.shape[0] == len(remain_index)
        for i in range(self.pred_y.shape[0]):
            pred_y[remain_index[i]] = self.pred_y[i]
        self.pred_y = torch.from_numpy(pred_y)

        return self.pred_y, self.final_output_embeddings

    def _loss(self):
        # 分类损失，采用sigmoid交叉熵损失
        self.class_loss = 0.0
        # print('labels', self.labels)
        # print('pred_logist', torch.squeeze(self.logist))
        logs = F.binary_cross_entropy_with_logits(self.logist.squeeze(), self.labels.float())
        self.class_loss = torch.mean(logs)

        # 嵌入损失，使同类型节点嵌入尽可能相似，不同类型节点嵌入尽可能远离
        self.graph_loss = 0.0
        pos_idx = torch.squeeze(torch.nonzero(self.labels == 1))
        pos_embeds = self.final_output_embeddings[pos_idx]  # 正样本的嵌入
        # print('size of pos_embeds', pos_embeds.size())
        if len(list(pos_embeds.size())) == 1:
            # print('pos_embeds is 1 dim')
            pos_embeds = torch.unsqueeze(pos_embeds, dim=0)
            # print('pos_embeds', pos_embeds.size())
        neg_idx = torch.squeeze(torch.nonzero(self.labels == 0))
        neg_embeds = self.final_output_embeddings[neg_idx]  # 负样本嵌入
        # print('size of neg_embeds', neg_embeds.size())
        if len(list(neg_embeds.size())) == 1:
            # print('neg_embeds is 1 dim')
            neg_embeds = torch.unsqueeze(neg_embeds, dim=0)
            # print('neg_embeds', neg_embeds.size())
        pos_score_1 = torch.mean(torch.matmul(pos_embeds, pos_embeds.permute(1, 0)), dim=1)
        pos_score_2 = torch.mean(torch.matmul(neg_embeds, neg_embeds.permute(1, 0)), dim=1)
        pos_score = torch.cat([pos_score_1, pos_score_2], dim=0) + 1e-5
        # print("shape of pos_score: ", pos_score)
        neg_score_1 = torch.mean((-1.0) * torch.matmul(pos_embeds, neg_embeds.permute(1, 0)), dim=1)
        neg_score_2 = torch.mean((-1.0) * torch.matmul(neg_embeds, pos_embeds.permute(1, 0)), dim=1)
        neg_score = torch.cat([neg_score_1, neg_score_2], dim=0) + 1e-5
        # print("shape of neg_score: ", neg_score)
        graph_logs_1 = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        graph_logs_2 = F.binary_cross_entropy_with_logits(neg_score, torch.ones_like(neg_score))
        # self.graph_loss = torch.mean(graph_logs_1) + torch.mean(graph_logs_2)

        self.graph_loss = graph_logs_1 + graph_logs_2

        # 正则化损失，鼓励网络权重的稀疏性，并防止模型过拟合
        self.reg_loss = 0.0
        for name, v in self.named_parameters():
            # print('all train_variables:', v.name)
            if "attention_layers" in name and "bias" not in name:
                # print('train_variables:', name, tf.reduce_mean(torch.norm(v, p=2) * cfg.weight_decay)
                self.reg_loss = self.reg_loss + torch.mean(torch.norm(v, p=2) * cfg.weight_decay)
        # print('self.reg_loss', self.reg_loss)
        self.loss = cfg.a * self.class_loss + cfg.b * self.graph_loss + cfg.g * self.reg_loss
        return self.loss, self.class_loss, self.graph_loss, self.reg_loss

    def _result_score(self):
        # remain_index = self.placeholders['remain_index']    # list
        batch_label = self.placeholders['batch_label']      # [N_t]tensor
        # pred_y = np.zeros(batch_label.shape[0])  # 初始化预测标签, [batchsize]array
        # # print('pred_y', self.pred_y)
        # # print('number of remain_index', remain_index)
        # # print('batch_label', batch_label)
        # assert self.pred_y.shape[0] == len(remain_index)
        # for i in range(self.pred_y.shape[0]):
        #     pred_y[remain_index[i]] = self.pred_y[i]
        # pred_y = torch.from_numpy(pred_y)
        accuracy = Accuracy()
        accuracy.update(self.pred_y, batch_label)
        accuracy = accuracy.compute().item()    # 获取准确率并转换为 NumPy 数组
        # print('acc', accuracy)
        recall = Recall()
        recall.update(self.pred_y, batch_label)
        recall = recall.compute().item()
        # print('recall', recall)
        precision = Precision()
        precision.update(self.pred_y, batch_label)
        precision = precision.compute().item()
        # print('precision', precision)
        # Auc = AUC()
        # Auc.update(pred_y, batch_label)
        # auc = Auc.compute().item()
        fpr, tpr, thresholds = roc_curve(batch_label, self.pred_y, pos_label=1)
        AUC = auc(fpr, tpr)
        # print('AUC', AUC)
        f1_score = 2.0 * ((precision * recall)/(precision + recall + 1e-9))
        # print('f1_score', f1_score)
        self.result_scores = [accuracy, recall, precision, AUC, f1_score]
        return self.result_scores



