# -*- coding: utf-8 -*-
# @Time    : 2024/5/6 10:32
# @Author  : chenlelan
# @File    : layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DefaultConfig

cfg = DefaultConfig()

class TemporalAttentionLayer(nn.Module):
    """ 时间注意力层 """
    def __init__(self, input_dim, out_dim, n_heads, max_time_steps, residual=False,
                 bias=True, use_position_embedding=True):
        super(TemporalAttentionLayer, self).__init__()

        self.input_dim = input_dim
        print("temporal input dim:", input_dim)
        self.output_dim = out_dim
        self.n_heads = n_heads
        self.max_time_steps = max_time_steps
        self.residual = residual
        self.bias = bias
        self.vars = {}
        self.Wq = nn.ModuleList()
        self.Wk = nn.ModuleList()
        self.Wv = nn.ModuleList()
        for j in range(self.n_heads):
            self.Wq.append(nn.Conv1d(in_channels=self.input_dim, out_channels=self.output_dim // self.n_heads, kernel_size=1))
            self.Wk.append(nn.Conv1d(in_channels=self.input_dim, out_channels=self.output_dim // self.n_heads, kernel_size=1))
            self.Wv.append(nn.Linear(self.input_dim, self.output_dim // self.n_heads))
        self.conv1d_layer = nn.Conv1d(in_channels=self.output_dim, out_channels=self.output_dim, kernel_size=1)
        xavier_init = nn.init.xavier_uniform_  # 权重初始化
        if use_position_embedding:
            self.position_embeddings = nn.Parameter(torch.empty(max_time_steps, self.input_dim), requires_grad=True)     # [T, F]
            xavier_init(self.position_embeddings)

    def forward(self, inputs):
        """ Computes multi-head temporal self-attention with positional embeddings."""

        # 1: Add position embeddings to input.
        position_inputs = torch.tile(torch.arange(inputs.size(1)).unsqueeze(0), [inputs.size(0), 1])    # [N, T]
        # inputs = inputs.float()
        # print('size of inputs', inputs.size())
        position_embedding = self.position_embeddings[position_inputs]
        # print('size of position_inputs', position_embedding.size())
        temporal_inputs = inputs + position_embedding  # [N, T, F]
        # print('size of temporal_inputs', temporal_inputs.size())

        # 2: Query, Key based multi-head self attention.
        q_all = []
        k_all = []
        v_all = []
        for j in range(self.n_heads):
            q = self.Wq[j](temporal_inputs.permute(0, 2, 1))  # [N, T, F'/h]
            q = q.permute(0, 2, 1)
            # print("temporal attention Q shape", q.size())
            k = self.Wk[j](temporal_inputs.permute(0, 2, 1))  # [N, T, F'/h]
            k = k.permute(0, 2, 1)
            v = self.Wv[j](temporal_inputs)  # [N, T, F'/h]
            q_all.append(q)
            k_all.append(k)
            v_all.append(v)

        # 3: concat and scale.
        q_ = torch.cat(q_all, dim=0)  # [hN, T, F/h]
        k_ = torch.cat(k_all, dim=0)  # [hN, T, F/h]
        v_ = torch.cat(v_all, dim=0)  # [hN, T, F/h]

        out = torch.matmul(q_, k_.transpose(1, 2))    # [hN, T, T]
        outputs = out / (k_.size(-1) ** 0.5)

        # 4：Masks
        diag_val = torch.ones_like(outputs[0, :, :])  # [T, T]
        tril = torch.tril(diag_val)     # [T, T],下三角矩阵
        masks = torch.tile(tril.unsqueeze(0), [outputs.size(0), 1, 1])  # [hN, T, T]
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        outputs = torch.where(masks == 0, padding, outputs)  # [h*N, T, T]
        outputs = F.softmax(outputs, dim=-1)  # Masked attention.

        # 5: Dropout on attention weights.
        # outputs = self.dropout_with_rate(outputs, rate=self.attn_drop)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]

        # 使用 torch.chunk() 函数将张量沿着指定维度分割成 self.n_heads 个张量
        split_outputs = torch.chunk(outputs, self.n_heads, dim=0)
        outputs = torch.cat(split_outputs, dim=-1)     # [N, T, F]

        # Optional: Feedforward and residual
        if cfg.position_ffn:
            outputs = self.feedforward(outputs)

        if self.residual:
            outputs += temporal_inputs

        return outputs

    def feedforward(self, inputs):
        """Point-wise feed forward net.
        """
        outputs = self.conv1d_layer(inputs.permute(0, 2, 1))
        outputs = outputs.permute(0, 2, 1)
        # 加入残差
        outputs += inputs
        return outputs
        
        


