# -*- coding:utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import collections
import copy
import pandas as pd
import numpy as np
import random
from time import time
from local_model import KGNN
from dp_mechanism import cal_sensitivity, Laplace, Gaussian_Simple, Gaussian_moment
from test import _ndcg_at_k
from test import _get_feed_label
from test import _get_topk_feed_data
from test import get_user_record
from data_loader import load_data
from data_loader import load_kg

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_auc_score

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, datasets

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def iid_partition(dataset, clients):
    """
    I.I.D.在客户机上对数据进行分割。
    洗牌数据
    分给客户
    params.Dataset
      - 数据集(Torch.utils.Dataset)。
      - 客户端（int）。要分割数据的客户数量
    返回。
      - 每个客户的索引字典
    """
    num_items_per_client = int(len(dataset) /clients)
    client_dict = {}
    idxs = [i for i in range(len(dataset))]
    for i in range(clients):
        client_dict[i] = set(np.random.choice(idxs, num_items_per_client, replace=False))
        idxs = list(set(idxs) - client_dict[i])
    return client_dict