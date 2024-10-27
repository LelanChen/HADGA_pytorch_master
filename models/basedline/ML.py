# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 17:41
# @Author  : chenlelan
# @File    : ML.py

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
from xgboost import XGBClassifier
import os
from datetime import datetime
from utils.write_to_csv import *
from config import DefaultConfig
from utils.preprocess import *
from models.result import get_result_matrix

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

def train_svm(X_train, Y_train):
    # 创建 SVM 模型
    clf = svm.SVC(kernel='linear')
    # 训练模型
    clf.fit(X_train, Y_train)
    # 保存模型
    joblib.dump(clf, MODEL_SAVE + 'svm_model.pkl')

def test_svm(X_test, Y_test):
    # 加载模型
    clf = joblib.load(MODEL_SAVE + 'svm_model.pkl')
    y_pred = clf.predict(X_test)
    test_results = get_result_matrix(Y_test, y_pred)
    output_name = CSV_SAVE + 'SVM_%s_%s_%s_%s.csv' % (cfg.dataset.split("/")[0], str(today.year),
                                                  str(today.month), str(today.day))
    write_eval_result(test_results, output_name, 'SVM', cfg.dataset, mod='test')


def train_RF(X_train, Y_train):
    # 创建随机森林分类器
    rf = RandomForestClassifier(n_estimators=100)
    # 训练模型
    rf.fit(X_train, Y_train)
    # 保存模型
    joblib.dump(rf, MODEL_SAVE + 'rf_model.pkl')

def test_RF(X_test, Y_test):
    rf = joblib.load(MODEL_SAVE + 'rf_model.pkl')
    y_pred = rf.predict(X_test)
    test_results = get_result_matrix(Y_test, y_pred)
    output_name = CSV_SAVE + 'RF_%s_%s_%s_%s.csv' % (cfg.dataset.split("/")[0], str(today.year),
                                                      str(today.month), str(today.day))
    write_eval_result(test_results, output_name, 'RF', cfg.dataset, mod='test')


def train_GBDT(X_train, Y_train):
    gbdt = GradientBoostingClassifier()
    gbdt.fit(X_train, Y_train)
    joblib.dump(gbdt, MODEL_SAVE + 'gbdt.pkl')

def test_GBDT(X_test, Y_test):
    gbdt = joblib.load(MODEL_SAVE + 'gbdt.pkl')
    y_pred = gbdt.predict(X_test)
    test_results = get_result_matrix(Y_test, y_pred)
    output_name = CSV_SAVE + 'GBDT_%s_%s_%s_%s.csv' % (cfg.dataset.split("/")[0], str(today.year),
                                                     str(today.month), str(today.day))
    write_eval_result(test_results, output_name, 'GBDT', cfg.dataset, mod='test')


def train_XGB(X_train, Y_train):
    xgb = XGBClassifier()
    xgb.fit(X_train, Y_train)
    joblib.dump(xgb, MODEL_SAVE + 'xgb.pkl')
    
def test_XGB(X_test, Y_test):
    xgb = joblib.load(MODEL_SAVE + 'xgb.pkl')
    y_pred = xgb.predict(X_test)
    test_results = get_result_matrix(Y_test, y_pred)
    output_name = CSV_SAVE + 'XGBoost_%s_%s_%s_%s.csv' % (cfg.dataset.split("/")[0], str(today.year),
                                                          str(today.month), str(today.day))
    write_eval_result(test_results, output_name, 'XGBoost', cfg.dataset, mod='test')

