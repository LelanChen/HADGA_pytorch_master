# -*- coding: utf-8 -*-
# @Time    : 2024/5/7 15:35
# @Author  : chenlelan
# @File    : classify.py

import logging
import os
from datetime import datetime
import torch
from utils.preprocess import *
from utils.tbatch import MinibatchIterator
from utils.write_to_csv import write_eval_result
from models.HADGA.model import HADGA
from config import DefaultConfig

cfg = DefaultConfig()

# 输入正确的基本模型和模型名称，以获取解析器文件夹
output_dir = "./logs/{}/".format(cfg.base_model)
LOG_DIR = output_dir + cfg.logs_dir + '/test_data/'
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
SAVE_DIR = output_dir + cfg.save_dir
CSV_DIR = output_dir + cfg.csv_dir + '/test_data/'
if not os.path.isdir(CSV_DIR):
    os.mkdir(CSV_DIR)
MODEL_DIR = output_dir + cfg.model_dir


datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
today = datetime.today()
# 训练日志记录格式：为每次运行创建一个新的日志目录。目录的默认名称是“log”，而<>.log的内容将每天附加 => 每天一个日志文件。
log_file = LOG_DIR + '/%s_%s_%s_%s.log' % (cfg.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day))

log_level = logging.INFO
logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

# 遍历 DefaultConfig 类的所有属性和值，并将它们保存到日志中
for attr, value in DefaultConfig.__dict__.items():
    if not attr.startswith('__') and not callable(value):
        logging.info(f'{attr}: {value}')

# 创建结果日志csv文件
output_file = CSV_DIR + '/%s_%s_%s_%s.csv' % (cfg.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day))
# 加载测试数据
graphs, adjs, labels, classes = load_graphs(cfg.test_dataset,'graphs_remap.npz')
node_features, edge_features = load_feats(cfg.test_dataset, "node_feat_remap.npz", "edge_feat_remap.npz")
num_features = len(node_features[0][0])  # 节点特征维度S
num_edge_features = len(list(edge_features[0].values())[0])  # 边缘特征维度E
testbatchIterator = MinibatchIterator(graphs, adjs, labels, classes, node_features, edge_features, num_features, num_edge_features)

# 加载模型
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
loaded_model = HADGA(num_features, num_edge_features, cfg.use_edge)
model_path = MODEL_DIR + "/" + "model_{}".format(cfg.dataset) + ".pth"
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.to(device)

batch_scores = []
val_scores = []
embedding_list = []
pred_label = []
remain_label = []

it = 0
while not testbatchIterator.end():
    print('****** eval_data batch {} ******'.format(it))
    test_feed_dict = testbatchIterator.next_batch_generator()
    test_feed_dict.update({'spatial_drop': 0.0})
    test_feed_dict.update({'temporal_drop': 0.0})
    pred_y, final_embedding = loaded_model(test_feed_dict)

    embedding_list.extend(final_embedding.tolist())
    pred_label.extend(pred_y)
    remain_label.extend(test_feed_dict['label'].to_list()[0])

    eval_scores = loaded_model._result_score()
    print('result:', eval_scores)
    batch_scores.append(eval_scores)
    it += 1

# 存储节点嵌入和预测标签
df = pd.DataFrame(embedding_list)
print("number of remain node:", len(remain_label))
df["Label"] = remain_label
df.to_csv(SAVE_DIR, index=False)

print('batch {} result：{}'.format(it, batch_scores))
val_scores = torch.mean(torch.Tensor(batch_scores), dim=0).numpy()
print('reduce mean result:', val_scores, type(val_scores), val_scores[3])

print("Val acc: {}, recall: {}, precision:{}, AUC:{}, f1_score:{}\n".format(
     val_scores[0], val_scores[1], val_scores[2], val_scores[3], val_scores[4]))

logging.info("Test results: acc: {}, recall: {}, precision: {}, AUC: {}, f1_score: {}\n".format(
    val_scores[0], val_scores[1], val_scores[2], val_scores[3], val_scores[4]))

write_eval_result(val_scores, output_file, cfg.base_model, cfg.dataset, mod='test_data')
