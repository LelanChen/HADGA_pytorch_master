# -*- coding: utf-8 -*-
# @Time    : 2024/5/6 9:46
# @Author  : chenlelan
# @File    : config.py

class DefaultConfig(object):
    # dataset path
    dataset = 'DDoS'
    train_dataset = dataset + "/train_data"
    eval_dataset = dataset + "/eval_data"
    test_dataset = dataset + "/test_data"
    class_num = 3   # numbers of classes

    # hyper-parameters
    epochs = 5
    batch_size = 5
    learning_rate = 0.001
    max_gradient_norm = 1.0     # Clip gradients to this norm.
    weight_decay = 0.005      # Weight for L2 loss on embedding matrix.
    spatial_drop = 0.2      # attn Dropout (1 - keep probability).
    temporal_drop = 0.2     # ffd Dropout rate (1 - keep probability.

    # model-set
    neighbor_sample_size = 4    # number of neighbors sampling.
    use_edge = False     # Whether to use edge information.
    use_residual = False    # Whether to use residual connections.
    neighbors_head_config = [4]      # Encoder layer config: attention heads in each neighbor attention layer.
    neighbors_layer_config = [32]    # Encoder layer config: units in each neighbor attention layer.
    temporal_head_config = [8, 4]   # Encoder layer config: attention heads in each temporal attention layer.
    temporal_layer_config = [64, 32]   # Encoder layer config: units in each temporal attention layer.
    position_ffn = True     # Whether to use position wise feedforward.
    window = 3    # Window size for temporal attention.

    # loss weight
    a = 1    # Weight for classification loss.
    b = 0.5  # Weight for graph loss.
    g = 0.5  # Weight for L2 loss.

    # eval_data set
    eval_freq = 2   # frequency of evaluation

    # opt-parameters
    optimizer = "adam"  # Optimizer for training: (adadelta, adam, rmsprop).
    seed = 123   # Random seed.

    # Directory structure
    base_model = "HADGA"    # Base model string.
    save_dir = "output"     # Save dir defaults to output/ within the base directory.
    logs_dir = "logs"  # Log dir defaults to logs/ within the base directory.
    csv_dir = "csv"         # CSV dir defaults to csv/ within the base directory.
    grad_dir = "grad_logs"
    model_dir = "model"     # Model dir defaults to model/ within the base directory.

# 实例
# cfg = DefaultConfig()
# print(cfg.window)