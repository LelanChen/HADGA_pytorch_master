# -*- coding: utf-8 -*-
# @Time    : 2024/4/21 16:11
# @Author  : chenlelan
# @File    : write_to_csv.py

import csv
import time
import numpy as np

def write_eval_result(test_results, output_name, model_name, dataset, mod='val'):
    """Output result scores to a csv file for result logging"""
    with open(output_name, 'a+') as f:
        writer = csv.writer(f)
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print('当前时间：', t)
        # writer.writerow("{},{},{},{},".format(dataset, model_name, mod, t))
        writer.writerow([dataset, model_name, mod, t])
        writer.writerow(["accuracy", "recall", "precision", "f1_score", "AUC"])  # 先写入列名
        print("{} results ({})".format(model_name, mod), test_results)
        writer.writerow(test_results)


def write_epoch_loss(epoch_loss, output_name, model_name, dataset, mod):
    with open(output_name, 'a+') as f:
        writer = csv.writer(f)
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        # print('当前时间：', t)
        writer.writerow([dataset, model_name, mod, t])
        writer.writerow(["epoch", "epoch_loss"])
        epoch = np.arange(len(epoch_loss))
        # print(epoch)
        train_loss = np.vstack((epoch, epoch_loss)).T
        # print(train_loss)
        writer.writerows(train_loss)
