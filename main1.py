# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import torch
import copy
import pandas as pd
from options import args_parser
from data_loader import load_data
from fedavg import FedAvg
from fedavg import iid_partition
from local_update import ClientUpdate
from test import testing
from train import train
from time import time
from local_model import KGNN

np.random.seed(555)


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def get_filepath(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.csv'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.csv'.format(log_count))
    return file_path


if __name__ == '__main__':
    # parse args
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    show_loss = False
    show_time = False
    show_topk = True

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    save_dir = 'log/{}_ldp/{}-{}epsilon/{}clip_lr{}_{}R_{}C_{}K_{}E_{}B_agg-{}_dim{}_l2{}_hop{}_neigh{}_batch{}/'.format(
        args.dataset, args.dp_mechanism, args.dp_epsilon, args.dp_clip, args.lr, args.rounds, args.frac, args.num_users, args.local_ep,
        args.local_bs, args.aggregator, args.dim, args.l2_weight, args.n_hop, args.neighbor_sample_size,
        args.batch_size)
    ensureDir(save_dir)
    args.log = get_filepath(save_dir)

    # LOAD DATA
    data_info = load_data(args)
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
    dp_epsilon = args.dp_epsilon
    dp_delta = args.dp_delta
    dp_mechanism = args.dp_mechanism
    dp_clip = args.dp_clip
    R = args.rounds
    K = args.num_users
    C = args.frac
    E = args.local_ep
    B = args.local_bs

    target_test_accuracy = 99.0
    # 用于训练的数据分区类型（IID）。
    iid_dict = iid_partition(train_data, 100)
    # root_path="../result/music/"
    root_path = "../result/{}_ldp/".format(args.dataset)
    plt_color = "orange"
    plt_title = 'FedKGR_{}-{}epsilon-{}clip on {}_{}R_{}C_{}K_{}E_{}B_'.format(args.dp_mechanism, args.dp_epsilon,
                                                                           args.dp_clip, args.dataset, args.rounds,
                                                                           args.frac,
                                                                           args.num_users, args.local_ep,
                                                                           args.local_bs)
    # load model
    model = KGNN(args, n_user, n_entity, n_relation, adj_entity, adj_relation)
    if torch.cuda.is_available():
        model.cuda()
    print(model)
    model.train()

    # copy weights得到全局模型权重参数
    w_glob = model.state_dict()
    # training loss and test accuracy
    train_Loss, test_Loss, test_Accuracy, test_AUC, test_F1, results_list = [], [], [], [], [], []
    curr_round = 0
    acc_best = 0
    auc_test,acc_test,f1_test = 0,0,0
    # measure time
    start = time()
    # for curr_round in range(1, R + 1):
    while 1:
        w_locals, local_loss = [], []
        local_auc, local_acc, local_f1 = [], [], []
        recall_list, ndcg_list = [], []
        curr_round+=1
        m = max(int(C * K), 1)
        S_t = np.random.choice(range(K), m, replace=False)
        # 客户端的训练
        time1 = time()
        for k in S_t:
            #       local_update = ClientUpdate(dataset=train_data_dict, batchSize=batch_size, learning_rate=lr, epochs=E, df=data_dict[k])
            local_update = ClientUpdate(args, dataset=train_data,
                                        idxs=iid_dict[k], dp_epsilon=dp_epsilon / (C * R),
                                        dp_delta=dp_delta,
                                        dp_mechanism=dp_mechanism, dp_clip=dp_clip)
            weights, loss = local_update.train(args, train_data, ripple_set, model=copy.deepcopy(model))

            w_locals.append(copy.deepcopy(weights))
            local_loss.append(copy.deepcopy(loss))

        # 更新全局权重 FedAvg算法聚合
        w_glob = FedAvg(w_locals)
        # 将更新后的权重参数移到我们的模型状态下
        model.load_state_dict(w_glob)

        # loss and auc acc f1
        loss_avg = sum(local_loss) / len(local_loss)

        # 全局测试
        model.eval()
        # test_loss, test_auc, test_acc, test_f1, recall, ndcg = testing(args, model, ripple_set, train_data, test_data)
        test_loss, test_auc, test_acc, test_f1= testing(args, model, ripple_set, train_data, test_data)
        time2 = time()  # 用于计算每一轮次的时间



        ctr_log = 'Round %d | test auc: %.4f  test acc: %.4f  test f1: %.4f' % (curr_round, test_auc, test_acc, test_f1)
        print(ctr_log)
        with open(args.log, 'a') as f:
            f.write(ctr_log + '\n')

        # topk_log = 'topk eval | recall: %s | ndcg: %s' % (
        #     (['%.4f' % r for r in recall]), ' '.join(['%.4f' % n for n in ndcg]))
        # print(topk_log)
        # with open(args.log, 'a') as f:
        #     f.write(topk_log + '\n')

        # for i in recall:
        #     recall_list.append(round(i, 4))
        # for i in ndcg:
        #     ndcg_list.append(round(i, 4))

        round_time = time2 - time1

        if test_acc > acc_best:
            acc_best = test_acc
            count = 0
        else:
            count += 1
        if count > 5:
            print('not improved for 10 epochs, stop trianing')
            break

        # results_list.append(
        #     [curr_round, round_time, loss_avg, test_loss, test_auc, test_acc, test_f1, recall_list, ndcg_list])
        results_list.append(
            [curr_round, round_time, loss_avg, test_loss, test_auc, test_acc, test_f1])

        # print(
        #     f"Round: {curr_round}... \tTime:{round_time}... \tAverage Train Loss: {round(loss_avg, 5)}... \tTest Loss: {round(test_loss, 5)}... \tTest AUC: {round(test_auc, 5)}... \tTest Accuracy: {round(test_acc, 5)}... \tTest F1: {round(test_f1, 5)}... \tRecall@: {recall_list}... \tNDCG@:{ndcg_list}")
        print(
            f"Round: {curr_round}... \tTime:{round_time}... \tAverage Train Loss: {round(loss_avg, 5)}... \tTest Loss: {round(test_loss, 5)}... \tTest AUC: {round(test_auc, 5)}... \tTest Accuracy: {round(test_acc, 5)}... \tTest F1: {round(test_f1, 5)}")



        # 供可视化用
        train_Loss.append(loss_avg)
        test_Loss.append(test_loss)
        test_AUC.append(test_auc)
        test_Accuracy.append(test_acc)
        test_F1.append(test_f1)

    # df = pd.DataFrame(data=results_list,
    #                   columns=["curr_round", "round time", "train loss_avg", "test loss", "test auc", "test acc",
    #                            "test f1",
    #                            "recall@", "ndcg@"])
    df = pd.DataFrame(data=results_list,
                      columns=["curr_round", "round time", "train loss_avg", "test loss", "test auc", "test acc",
                               "test f1"])

    df.to_csv(root_path + 'FedKGR_{}-epsilon{}-clip{} on {}_{}R_{}C_{}K_{}E_{}B.csv'.format(
        args.dp_mechanism, args.dp_epsilon, args.dp_clip, args.dataset, R, C, K, E, args.local_bs), index=None)

    end = time()

    # plot Loss
    fig, ax = plt.subplots()
    x_axis = np.arange(1, R + 1)
    y_axis = np.array(train_Loss)
    ax.plot(x_axis, y_axis, 'tab:' + plt_color)
    ax.set(xlabel='Number of Rounds', ylabel='Train Loss',
           title='FedKGR-{} on {}'.format(args.dp_mechanism, args.dataset))
    ax.grid()
    # fig.savefig(root_path + plt_title + 'trainloss.png')

    fig0, ax0 = plt.subplots()
    x_axis0 = np.arange(1, curr_round + 1)
    y_axis0 = np.array(test_Loss)
    ax0.plot(x_axis0, y_axis0, 'tab:' + plt_color)
    ax0.set(xlabel='Number of Rounds', ylabel='Test Loss',
            title='FedKGR-{} on {}'.format(args.dp_mechanism, args.dataset))
    ax0.grid()
    # fig0.savefig(root_path + plt_title + 'testloss.png')

    # plot test auc
    fig1, ax1 = plt.subplots()
    x_axis1 = np.arange(1, curr_round + 1)
    y_axis1 = np.array(test_AUC)
    ax1.plot(x_axis1, y_axis1)
    ax1.set(xlabel='Number of Rounds', ylabel='AUC', title='FedKGR-{} on {}'.format(args.dp_mechanism, args.dataset))
    ax1.grid()
    # fig1.savefig(root_path + plt_title + 'test_auc.png')

    # plot test accuracy
    fig2, ax2 = plt.subplots()
    x_axis2 = np.arange(1, curr_round + 1)
    y_axis2 = np.array(test_Accuracy)
    ax2.plot(x_axis2, y_axis2)
    ax2.set(xlabel='Number of Rounds', ylabel='Accuracy',
            title='FedKGR-{} on {}'.format(args.dp_mechanism, args.dataset))
    ax2.grid()
    # fig2.savefig(root_path + plt_title + 'test_acc.png')

    # plot test f1
    fig3, ax3 = plt.subplots()
    x_axis3 = np.arange(1, curr_round + 1)
    y_axis3 = np.array(test_F1)
    ax3.plot(x_axis3, y_axis3)
    ax3.set(xlabel='Number of Rounds', ylabel='F1', title='FedKGR-{} on {}'.format(args.dp_mechanism, args.dataset))
    ax3.grid()
    # fig3.savefig(root_path + plt_title + 'test_f1.png')

    print("Training Done!")
    print("Total time taken to Train: {}".format(end - start))



# train(args, data_info, show_loss, show_topk)
# train(args, data_info, show_loss)
# if show_time:
#     print('time used: %d s' % (time() - t))


# music_kgnn_iid_trained = training(args, model, args.rounds, args.local_bs, args.lr, train_data, iid_dict, test_data,
#                                   args.frac,
#                                   args.num_users, args.local_ep, plt_title, "orange", target_test_accuracy, root_path)
# torch.save(music_kgnn_iid_trained, root_path + "FedKGR_dp-{} on IID {}_{}R_{}C_{}K_{}E_{}B_epsilon-{}.pth".format(
#     args.dp_mechanism, args.dataset, args.rounds, args.frac, args.num_users, args.local_ep, args.local_bs,
#     args.dp_epsilon))
