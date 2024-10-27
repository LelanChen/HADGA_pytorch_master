# -*- coding: utf-8 -*-
# @Time    : 2024/5/29 10:42
# @Author  : chenlelan
# @File    : result.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, f1_score

def get_result_matrix(label, y_pred):
    # 计算准确率
    accuracy = accuracy_score(label, y_pred)
    # 计算召回率
    recall = recall_score(label, y_pred)
    # 计算精度
    precision = precision_score(label, y_pred)
    # AUC
    fpr, tpr, thresholds = roc_curve(label, y_pred, pos_label=1)
    AUC = auc(fpr, tpr)
    # 计算 F1-score
    f1 = f1_score(label, y_pred)

    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("AUC:", AUC)
    print("F1-score:", f1)
    result_scores = [accuracy, recall, precision, AUC, f1]
    return result_scores