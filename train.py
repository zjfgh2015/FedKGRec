# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import torch

# from model import RippleNet
from local_model import KGNN
from test import ctr_eval
from test import topk_eval
from test import _get_topk_feed_data
from test import get_user_record
from test import get_feed_dict



def train(args, data_info, show_loss):  # train方法需要用到data info是从main的load data方法里来的
# def train(args, data_info, show_loss, show_topk):
    n_user = data_info[0]
    n_item = data_info[1]
    train_data = data_info[2]
    eval_data = data_info[3]
    test_data = data_info[4]
    n_entity = data_info[5]
    n_relation = data_info[6]
    ripple_set = data_info[7]
    adj_entity = data_info[8]
    adj_relation = data_info[9]

    model = KGNN(args, n_user, n_entity, n_relation, adj_entity, adj_relation) # 在train里用到model
    print(model)

    # top-K evaluation settings
    # user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item)
    if args.use_cuda:
        model.cuda()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,weight_decay=args.l2_weight
    )

    results_list = []
    for step in range(args.n_epoch):
        # training
        np.random.shuffle(train_data)
        start = 0
        while start < train_data.shape[0]:
            return_dict = model(*get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
            loss = return_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            start += args.batch_size
            if show_loss:
                print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss.item()))

        # CTR evaluation
        train_auc, train_acc, train_f1 = ctr_eval(args, model, train_data, ripple_set, args.batch_size)
        eval_auc, eval_acc, eval_f1 = ctr_eval(args, model, eval_data, ripple_set, args.batch_size)
        test_auc, test_acc, test_f1 = ctr_eval(args, model, test_data, ripple_set, args.batch_size)

        print('epoch %d    train auc: %.4f  acc: %.4f f1: %.4f    eval auc: %.4f  acc: %.4f f1: %.4f    test auc: %.4f  acc: %.4f f1: %.4f'
              % (step, train_auc, train_acc, train_f1, eval_auc, eval_acc, eval_f1, test_auc, test_acc, test_f1))

# top-K evaluation
        # if show_topk:
        #     precision, recall = topk_eval(
        #         args, model, user_list, train_record, test_record, item_set, k_list, args.batch_size)
        #     print('precision: ', end='')
        #     for i in precision:
        #         print('%.4f\t' % i, end='')
        #     print()
        #     print('recall: ', end='')
        #     for i in recall:
        #         print('%.4f\t' % i, end='')
        #     print('\n')

        # results_list.append([step, train_auc, train_acc, train_f1, eval_auc, eval_acc, eval_f1, test_auc, test_acc, test_f1, precision, recall])
        results_list.append([step, train_auc, train_acc, train_f1, eval_auc, eval_acc, eval_f1, test_auc, test_acc, test_f1])

    df=pd.DataFrame(data=results_list,columns=["epoch", "train auc", "train acc", "train f1", "eval auc", "eval acc", "eval f1", "test auc", "test acc", "test f1"])
    # df.to_csv('../result/KGNN-book2.csv',index=None)
    # df.to_csv('../result/KGNN-movie-1m2.csv',index=None)
    # df.to_csv('../result/KGNN-movie-20m.csv',index=None)
    # df.to_csv('../result/KGNN-music3.csv',index=None)
    df.to_csv('../result/KGNN-restaurant2.csv',index=None)





