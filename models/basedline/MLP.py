# -*- coding: utf-8 -*-
# @Time    : 2024/5/29 9:58
# @Author  : chenlelan
# @File    : MLP.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from datetime import datetime
from utils.write_to_csv import *
from models.result import get_result_matrix
from config import DefaultConfig

cfg = DefaultConfig

OUT_DIR = '../../logs/DL/'
if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)
CSV_SAVE = OUT_DIR + 'csv/'
if not os.path.isdir(CSV_SAVE):
    os.mkdir(CSV_SAVE)
MODEL_SAVE = OUT_DIR + 'model/'
if not os.path.isdir(MODEL_SAVE):
    os.mkdir(MODEL_SAVE)

datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
today = datetime.today()

class MLP(nn.Module):
    def __init__(self, num_feature):
        super(MLP, self).__init__()
        self.input_dim = num_feature
        self.layer1 = nn.Linear(self.input_dim, 24)  # 输入维度为 100
        self.layer2 = nn.Linear(24, 16)
        self.layer3 = nn.Linear(16, 8)
        self.layer4 = nn.Linear(8, 2)  # 二分类输出维度为 2

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x), negative_slope=0.02)
        x = F.leaky_relu(self.layer2(x), negative_slope=0.02)
        x = F.leaky_relu(self.layer3(x), negative_slope=0.02)
        x = self.layer4(x)
        return x

def train_MLP(X_train, Y_train, X_eval, Y_eval):
    num_feature = X_train.shape[1]
    model = MLP(num_feature)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    eval_auc = []
    # 训练模型
    for epoch in range(200):
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 评估模型
        if epoch % cfg.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_eval)
                _, y_pred = torch.max(test_outputs, 1)
            eval_results = get_result_matrix(Y_eval, y_pred)
            print("epoch {} eval result: acc{}, recall{}, precision{}, AUC{}, f1-score{}".format(epoch,
                    eval_results[0], eval_results[1], eval_results[2], eval_results[3], eval_results[4],))
            auc = eval_results[3]
            eval_auc.append(auc)
            if (epoch == 0) or (epoch > 0 and auc >= max(eval_auc)):
                # 保存模型
                torch.save(model.state_dict(), MODEL_SAVE + 'mlp.pth')

def test_MLP(X_test, Y_test):
    num_feature = X_test.shape[1]
    model = MLP(num_feature)
    model.load_state_dict(torch.load(MODEL_SAVE + 'mlp.pth'))
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, y_pred = torch.max(test_outputs, 1)
    test_results = get_result_matrix(Y_test, y_pred)
    output_name = CSV_SAVE + 'MLP_%s_%s_%s_%s.csv' % (cfg.dataset.split("/")[0], str(today.year),
                                                      str(today.month), str(today.day))
    write_eval_result(test_results, output_name, 'MLP', cfg.dataset, mod='test')

