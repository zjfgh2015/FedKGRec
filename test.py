# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

def testing(args, model, ripple_set, train_data, test_data):
    # test loss
    #   dataset=KGCNplusDataset(dataset,df)
    model.eval()
    test_loader = DataLoader(test_data, batch_size=args.bs)
    if args.use_cuda:
        model.cuda()
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    test_loss = 0.0
    result_recall_list = []
    result_ndcg_list = []
    test_auc, test_acc, test_f1 = [],[],[]
    start = 0
    # np.random.shuffle(test_data)
    # for idx, data in enumerate(test_loader):
    while start < test_data.shape[0]:
    #     if torch.cuda.is_available():
    #         data, labels = data.cuda(), labels.cuda()
            return_dict = model(*get_feed_dict(args, model, test_data, ripple_set, start, start + args.batch_size))
            loss = return_dict["loss"]
            scores = return_dict["scores"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            test_loss += loss.item()
            start += args.bs
            # auc, acc, f1 = ctr_eval(args, model, data, ripple_set, args.batch_size)
            # test_auc.append(auc)
            # test_acc.append(acc)
            # test_f1.append(f1)

    # CTR evaluation
    test_auc, test_acc, test_f1 = ctr_eval(args, model, test_data, ripple_set, args.batch_size)
    # test_AUC = test_auc.sum()/len(test_loader.dataset)
    # test_ACC = test_acc.sum()/len(test_loader.dataset)
    # test_F1 = test_f1.sum()/len(test_loader.dataset)

    # Top-K evaluation
    if args.show_topk:
        recall, ndcg = topk_eval(args, model, train_data, test_loader.dataset, ripple_set, args.batch_size)

    # avg test loss and accuracy
    avg_loss = test_loss / len(test_loader.dataset)
    #   print("Test Loss: {:.6f}\n".format(test_loss))

    # return avg_loss, test_auc, test_acc, test_f1, recall, ndcg
    return avg_loss, test_auc, test_acc, test_f1


def _get_feed_label(args, labels):
    labels = torch.FloatTensor(labels)
    if args.use_cuda:
        labels = labels.cuda()
    return labels

def get_feed_dict_new(args, model, data, ripple_set): #模型输入一个batch的 user item
    users = data[0]
    items = data[1]
    labels = data[2]
    memories_h, memories_r, memories_t = [], [], [] #[user]
    for i in range(args.n_hop):
        memories_h.append(torch.LongTensor([ripple_set[user][i][0] for user in users.tolist()]))
        memories_r.append(torch.LongTensor([ripple_set[user][i][1] for user in users.tolist()]))
        memories_t.append(torch.LongTensor([ripple_set[user][i][2] for user in users.tolist()]))
    if args.use_cuda:
        users = users.cuda()
        items = items.cuda()
        labels = labels.cuda()
        memories_h = list(map(lambda x: x.cuda(), memories_h))
        memories_r = list(map(lambda x: x.cuda(), memories_r))
        memories_t = list(map(lambda x: x.cuda(), memories_t))
    return users, items, labels, memories_h, memories_r, memories_t

def get_feed_dict(args, model, data, ripple_set, start, end): #模型输入一个batch的 user item
    users = torch.LongTensor(data[start:end, 0])
    items = torch.LongTensor(data[start:end, 1])
    labels = torch.LongTensor(data[start:end, 2])
    memories_h, memories_r, memories_t = [], [], [] #[user]
    for i in range(args.n_hop):
        memories_h.append(torch.LongTensor([ripple_set[user][i][0] for user in data[start:end, 0]]))
        memories_r.append(torch.LongTensor([ripple_set[user][i][1] for user in data[start:end, 0]]))
        memories_t.append(torch.LongTensor([ripple_set[user][i][2] for user in data[start:end, 0]]))
    if args.use_cuda:
        users = users.cuda()
        items = items.cuda()
        labels = labels.cuda()
        memories_h = list(map(lambda x: x.cuda(), memories_h))
        memories_r = list(map(lambda x: x.cuda(), memories_r))
        memories_t = list(map(lambda x: x.cuda(), memories_t))
    return users, items, labels, memories_h, memories_r, memories_t

#点击预测评价函数
def ctr_eval(args, model, data, ripple_set, batch_size):#评价函数
    start = 0
    auc_list = []
    acc_list = []
    f1_list = []
    # model.eval()
    while start < data.shape[0]:
        auc, acc, f1 = model.evaluate(*get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        f1_list.append(f1)
        start += batch_size
    # model.train()
    return float(np.mean(auc_list)), float(np.mean(acc_list)), float(np.mean(f1_list))


# def topk_settings(show_topk, train_data, test_data, n_item):
#     if show_topk:
#         user_num = 100
#         k_list = [1, 2, 5, 10, 20, 50, 100]
#         train_record = get_user_record(train_data, True)
#         test_record = get_user_record(test_data, False)
#         user_list = list(set(train_record.keys()) & set(test_record.keys()))
#         if len(user_list) > user_num:
#             user_list = np.random.choice(user_list, size=user_num, replace=False)
#         item_set = set(list(range(n_item)))
#         return user_list, train_record, test_record, item_set, k_list
#     else:
#         return [None] * 5
#
# def topk_eval(args, model, user_list, train_record, test_record, item_set, k_list, batch_size):
#     precision_list = {k: [] for k in k_list}
#     recall_list = {k: [] for k in k_list}
#     for user in user_list:
#         test_item_list = list(item_set - train_record[user])
#         item_score_map = dict()
#         start = 0
#         while start + batch_size <= len(test_item_list):
#             items, scores = model.get_scores(args, {model.user_indices: [user] * batch_size,
#                                                     model.item_indices: test_item_list[start:start + batch_size]})
#             for item, score in zip(items, scores):
#                 item_score_map[item] = score
#             start += batch_size
#         # padding the last incomplete minibatch if exists
#         if start < len(test_item_list):
#             items, scores = model.get_scores(
#                 args, {model.user_indices: [user] * batch_size,
#                        model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
#                                batch_size - len(test_item_list) + start)})
#             for item, score in zip(items, scores):
#                 item_score_map[item] = score
#         item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
#         item_sorted = [i[0] for i in item_score_pair_sorted]
#         for k in k_list:
#             hit_num = len(set(item_sorted[:k]) & test_record[user])
#             precision_list[k].append(hit_num / k)
#             recall_list[k].append(hit_num / len(test_record[user]))
#     precision = [np.mean(precision_list[k]) for k in k_list]
#     recall = [np.mean(recall_list[k]) for k in k_list]
#     return precision, recall

def topk_eval(args, model, train_data, test_data, ripple_set, batch_size):
    # logging.info('calculating recall ...')
    user_num = 100
    k_list = [1, 2, 5, 10, 20, 50, 100]
    #     precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    #     f1_list = {k: [] for k in k_list}  # f1
    ndcg_list = {k: [] for k in k_list}  # ndcg
    item_set = set(train_data[:,1].tolist() + test_data[:,1].tolist())
    # item_set = set(list(range(n_item)))
    train_record = get_user_record(train_data, True)
    test_record = get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    if len(user_list) > user_num:
        np.random.seed()
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    #     model.eval()
    for user in user_list:
        test_item_list = list(item_set - set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + args.batch_size <= len(test_item_list):
            items = test_item_list[start:start + args.batch_size]
            labels = [1] * batch_size
            input_data = _get_topk_feed_data(args, user, items, labels)
            scores = model(*get_feed_dict(args, model, input_data, ripple_set, 0, args.batch_size))
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += args.batch_size
        # padding the last incomplete mini-batch if exists
        if start < len(test_item_list):
            res_items = test_item_list[start:] + [test_item_list[-1]] * (args.batch_size - len(test_item_list) + start)
            labels = [1] * batch_size
            input_data = _get_topk_feed_data(args, user, res_items, labels)
            scores = model(*get_feed_dict(args, model, input_data, ripple_set, 0, args.batch_size))
            for item, score in zip(res_items, scores):
                item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        for k in k_list:
            topk_items_list = item_sorted[:k]
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
            #             precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(set(test_record[user])))
            #             f1_score = (2 * hit_num) / (len(set(test_record[user])) + k)
            #             f1_list[k].append(f1_score)
            topk_items = list(set(topk_items_list))
            topk_items.sort(key=topk_items_list.index)
            ndcg_list[k].append(_ndcg_at_k(k, topk_items, list(set(test_record[user]))))
    #     model.train()
    #     precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    #     f1 = [np.mean(f1_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]
    #     return precision, recall, f1, ndcg
    return recall, ndcg

def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

def _get_topk_feed_data(args, user, items, labels):
    res = list()
    # labels = torch.FloatTensor(labels)
    # if args.use_cuda:
    #     labels = labels.cuda()
    for item in items:
        for label in labels:
            res.append([user, item, label])
    return np.array(res)

def dcg_at_k(r, k):
    """Score is discounted cumulative gain (dcg)
    Relevance is binary (nonzero is relevant).
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    dcg = np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return dcg

# def _ndcg_at_k(r, k):
#     """Score is normalized discounted cumulative gain (ndcg)
#     Relevance is binary (nonzero is relevant).
#     Returns:
#         Normalized discounted cumulative gain
#     """
#     assert k >= 1
#     idcg = dcg_at_k(sorted(r, reverse=True), k)
#     if not idcg:
#         return 0.
#     return dcg_at_k(r, k) / idcg

def _ndcg_at_k(k, topk_items, test_items):
    dcg = 0
    for i in range(k):
        if len(topk_items) > i:
            if topk_items[i] in test_items:
                dcg += (2 ** 1 - 1) / np.log2(i + 2)
        else:
            "handle the case when topk_items is too short"
    idcg = 0
    n = len(test_items) if len(test_items) < k else k
    for i in range(n):
        idcg += (2 ** 1 - 1) / np.log2(i + 2)
    if dcg == 0 or idcg == 0:
        return 0
    return dcg / idcg

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def recall_at_k(r, k, all_pos_num):
    """Score is recall @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Recall @ k
    """
    assert k >= 1
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num