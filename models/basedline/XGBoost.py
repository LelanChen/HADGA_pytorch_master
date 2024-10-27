# -*- coding: utf-8 -*-
# @Time    : 2024/5/29 15:07
# @Author  : chenlelan
# @File    : XGBoost.py

import xgboost as xgb
from xgboost import plot_importance
from models.result import get_result_matrix
import pickle
import os
from datetime import datetime
from config import DefaultConfig
from utils.write_to_csv import write_eval_result

cfg = DefaultConfig

OUT_DIR = '../../logs/ML/'
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

# set XGBoost's parameters
params = {
    # 通用参数
    'booster': 'gbtree',
    'nthread': 4,
    # 'silent': 1,
    'seed': 123,
    # 任务参数
    # 'objective': 'multi:softmax',  # 多分类问题
    'objective': 'binary:logistic',     # 二分类
    'num_class': 2,  #类别总数
    # 提升参数
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'eta': 0.1,
}

def train_XGB(X_train, Y_train, X_eval, Y_eval):
    plst = params.items()
    plst = list(params.items())
    dtrain = xgb.DMatrix(X_train, Y_train)
    num_rounds = 500
    best_auc = 0
    for round in range(num_rounds):
        model = xgb.train(plst, dtrain, num_boost_round=1)
        deval = xgb.DMatrix(X_eval)
        y_pred = model.predict(deval)
        eval_results = get_result_matrix(Y_eval, y_pred)
        auc = eval_results[3]
        if auc > best_auc:
            best_auc = auc
            best_round = round
            best_model = model
    # 保存最佳模型
    with open(MODEL_SAVE + 'XGBoost.pkl', 'wb') as f:
        pickle.dump(best_model, f)

def test_XGB(X_test, Y_test):
    # 加载模型
    with open(MODEL_SAVE + 'XGBoost.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    dtest = xgb.DMatrix(X_test)
    y_pred = loaded_model.predict(dtest)
    test_results = get_result_matrix(Y_test, y_pred)
    output_name = CSV_SAVE + 'XGBoost_%s_%s_%s_%s.csv' % (cfg.dataset.split("/")[0], str(today.year),
                                                      str(today.month), str(today.day))
    write_eval_result(test_results, output_name, 'XGBoost', cfg.dataset, mod='test')