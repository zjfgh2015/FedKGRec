# -*- coding:utf-8 -*-
import copy
import pandas as pd
import numpy as np
import random
from time import time
from local_model import KGNN
from dp_mechanism import cal_sensitivity, Laplace, Gaussian_Simple, Gaussian_moment
from test import _ndcg_at_k
from test import get_feed_dict_new


import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, datasets


# 自定义数据类
class KGNNDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        # user_id = np.array(self.df.iloc[idx]['userID'])
        # item_id = np.array(self.df.iloc[idx]['itemID'])
        # label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        user_id, item_id, label = self.dataset[self.idxs[item]]
        return user_id, item_id, label


class ClientUpdate(object):
    def __init__(self, args, dataset, idxs, dp_mechanism='Laplace', dp_epsilon=0.1, dp_delta=1e-5, dp_clip=0.0005):
        self.args = args
        self.train_loader = DataLoader(KGNNDataset(dataset, idxs), batch_size=args.local_bs,shuffle=True)
        # self.learning_rate = learning_rate
        # self.epochs = epochs
        self.idxs = idxs
        self.dp_mechanism = args.dp_mechanism
        self.dp_epsilon = args.dp_epsilon
        self.dp_delta = args.dp_delta
        self.dp_clip = args.dp_clip

    def train(self, args, train_data, ripple_set, model):
        model.train()
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        #     criterion = nn.CrossEntropyLoss()
        #     optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.5)
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.BCELoss()
        if args.use_cuda:
            model.cuda()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.lr, weight_decay=args.l2_weight
        )
        results_list = []
        e_loss = []

        for step in range(args.local_ep):
            # training
            train_loss = 0.0
            t0 = time()
            np.random.shuffle(train_data)
            start = 0
            for batch_idx, data in enumerate(self.train_loader):
                # data = torch.tensor(data).to(self.args.device)
                # while start < data[0].shape[0]:
                model.zero_grad()
                return_dict = model(*get_feed_dict_new(args, model, data, ripple_set))
                loss = return_dict["loss"]
                loss.backward()
                if self.dp_mechanism != 'no_dp':
                    self.clip_gradients(model)
                optimizer.step()
                # scheduler.step()
                train_loss += loss.item()
                # start += args.batch_size
            train_time = time() - t0

            # average losses
            train_loss = train_loss / len(self.train_loader.dataset)
            e_loss.append(train_loss)

            print('epoch %d  train time: %.5f  train loss: %.5f  '
                  % (step, train_time, train_loss))

        # add noises to parameters
        if self.dp_mechanism != 'no_dp':
            self.add_noise(model)
        total_loss = sum(e_loss) / len(e_loss)
        return model.state_dict(), total_loss

    def clip_gradients(self, model):
        if self.dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            for k, v in model.named_parameters():
                # v.grad /= max(1, v.grad.norm(1) / self.dp_clip)
                try:
                    v.grad /= max(1, v.grad.norm(1) / self.dp_clip)
                except AttributeError:
                    "handle the case when v.grad is None"
        elif self.dp_mechanism == 'Gaussian':
            # Gaussian use 2 norm
            for k, v in model.named_parameters():
                # v.grad /= max(1, v.grad.norm(2) / self.dp_clip)
                try:
                    v.grad /= max(1, v.grad.norm(2) / self.dp_clip)
                except AttributeError:
                    "handle the case when v.grad is None"

    def add_noise(self, model):
        sensitivity = cal_sensitivity(self.args.lr, self.dp_clip, len(self.idxs))
        if self.dp_mechanism == 'Laplace':
            with torch.no_grad():
                for k, v in model.named_parameters():
                    noise = Laplace(epsilon=self.dp_epsilon, sensitivity=sensitivity, size=v.shape)
                    noise = torch.from_numpy(noise).to(self.args.device)
                    v += noise
        elif self.dp_mechanism == 'Gaussian':
            with torch.no_grad():
                for k, v in model.named_parameters():
                    noise = Gaussian_Simple(epsilon=self.dp_epsilon, delta=self.dp_delta, sensitivity=sensitivity,
                                            size=v.shape)
                    noise = torch.from_numpy(noise).to(self.args.device)
                    v += noise

