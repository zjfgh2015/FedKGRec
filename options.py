# -*- coding:utf-8 -*-
import argparse
import numpy as np
import os
np.random.seed(555)


def args_parser():
    # default settings for movie-1m
    # 1.后缀output1-文中给的参数是dim 32; n_hop 2; kge_w 0.01;l2_w 1e-7;lr 0.02;batch_size 1024;n_epoch 30;n_memory 32;neighbor_sample_size 4
    # 2.后缀output-代码原始参数：dim 8; n_hop 2; kge_w 0.01;l2_w 1e-7;lr 0.02;batch_size 512;n_epoch 30;n_memory 32;neighbor_sample_size 8
    # 3.后缀不加output-文中给的参数效果更好
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='movie-1m', help='which dataset to use')
    parser.add_argument('--dim', type=int, default=32, help='dimension of entity and relation embeddings')
    parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--bs', type=int, default=512, help='testing batch size') #用于testing的test_loader
    parser.add_argument('--n_epoch', type=int, default=30, help='the number of epochs')
    parser.add_argument('--n_memory', type=int, default=32, help='numbers of each ripple set')
    parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                        help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,
                        help='whether using outputs of all hops or just the last hop when making prediction')
    parser.add_argument('--n_heads', type=int, default=1, help='heads of self attention')
    parser.add_argument('--feed_f_dim', type=int, default=16, help='dim of feed forward network in transformer')
    parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
    parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--show_topk', type=bool, default=False, help='whether showing topk or not')
    parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')

    parser.add_argument('--rounds', type=int, default=200, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.05, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")  # 只用来计算了train_loader
    parser.add_argument('--dp_mechanism', type=str, default='Laplace', help='differential privacy mechanism')
    parser.add_argument('--dp_epsilon', type=float, default=10, help='differential privacy epsilon  0.1/1/2/5/10')
    parser.add_argument('--dp_delta', type=float, default=1e-5, help='differential privacy delta')
    parser.add_argument('--dp_clip', type=float, default=0.005, help='differential privacy clip 0.001/0.005/0.0005')
    '''

    # default settings for movie-20m
    # 1.后缀output-代码原始参数是dim 32; n_hop 2; kge_w 0.01;l2_w 1e-7;lr 0.02;batch_size 2048;n_epoch 4;n_memory 32;neighbor_sample_size 4
    # 2.后缀不加output-无文中给的参数
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='movie-20m', help='which dataset to use')
    parser.add_argument('--dim', type=int, default=32, help='dimension of entity and relation embeddings')
    parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=30, help='the number of epochs')
    parser.add_argument('--n_memory', type=int, default=32, help='numbers of each ripple set')
    parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                        help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,
                        help='whether using outputs of all hops or just the last hop when making prediction')
    parser.add_argument('--n_heads', type=int, default=1, help='heads of self attention')
    parser.add_argument('--feed_f_dim', type=int, default=16, help='dim of feed forward network in transformer')
    parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
    parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
    '''

    # default settings for Book-Crossing
    # 1.后缀output1-文中给的参数是dim 64; n_hop 2; kge_w 1e-2;l2_w 2e-5;lr 1e-4;batch_size 512;n_epoch 30;n_memory 32;neighbor_sample_size 8
    # 2.后缀output-代码原始参数是dim 4; n_hop 2; kge_w 1e-2;l2_w 1e-5;lr 1e-2;batch_size 1024;n_epoch 30;n_memory 32;neighbor_sample_size 4
    # 3.后缀不加output-原始参数效果更好
    # 4.后缀是book2-调整了模型用的model_new
    # '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
    parser.add_argument('--dim', type=int, default=4, help='dimension of entity and relation embeddings')
    parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--kge_weight', type=float, default=1e-2, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--bs', type=int, default=512, help='testing batch size') #用于testing的test_loader
    parser.add_argument('--n_epoch', type=int, default=30, help='the number of epochs')
    parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
    parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                        help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,
                        help='whether using outputs of all hops or just the last hop when making prediction')
    parser.add_argument('--n_heads', type=int, default=1, help='heads of self attention')
    parser.add_argument('--feed_f_dim', type=int, default=16, help='dim of feed forward network in transformer')
    parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
    parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--show_topk', type=bool, default=False, help='whether showing topk or not')
    parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')

    parser.add_argument('--rounds', type=int, default=200, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.05, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")  # 只用来计算了train_loader
    parser.add_argument('--dp_mechanism', type=str, default='Laplace', help='differential privacy mechanism')
    parser.add_argument('--dp_epsilon', type=float, default=0.1, help='differential privacy epsilon  0.1/1/2/5/10')
    parser.add_argument('--dp_delta', type=float, default=1e-5, help='differential privacy delta')
    parser.add_argument('--dp_clip', type=float, default=0.001, help='differential privacy clip 0.001/0.005/0.0005')
    # '''

    # default settings for lastfm 2k
    # 1.后缀output1-KGCN文中给的参数是dim 16; n_hop 2; kge_w 0.01;l2_w 1e-4;lr 5e-4;batch_size 512;n_epoch 30;n_memory 32;neighbor_sample_size 8
    # 2.后缀output-代码原始参数：dim 8; n_hop 2; kge_w 0.01;l2_w 1e-7;lr 0.02;batch_size 512;n_epoch 30;n_memory 32;neighbor_sample_size 8
    # 3.后缀不加output-原始参数效果更好
    # 4.后缀是music1-原始KGCN-tensorflow代码参数 dim 16; n_hop 2; kge_w 0.01;l2_w 1e-4;lr 5e-4;batch_size 128;n_epoch 30;n_memory 32;neighbor_sample_size 8
    # 5.后缀是music2-调整了模型用的model_new
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
    parser.add_argument('--dim', type=int, default=8, help='dimension of entity and relation embeddings')
    parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--kge_weight', type=float, default=0.001, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size') #用于本地模型计算
    parser.add_argument('--bs', type=int, default=512, help='testing batch size') #用于testing的test_loader
    # parser.add_argument('--n_layer', type=int, default=1, help='depth of layer')
    parser.add_argument('--n_epoch', type=int, default=30, help='the number of epochs')
    parser.add_argument('--n_memory', type=int, default=32, help='numbers of each ripple set')
    parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                        help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,
                        help='whether using outputs of all hops or just the last hop when making prediction')
    parser.add_argument('--n_heads', type=int, default=1, help='heads of self attention')
    parser.add_argument('--feed_f_dim', type=int, default=16, help='dim of feed forward network in transformer')
    parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
    parser.add_argument('--n_iter', type=int, default=2,
                        help='number of iterations when computing entity representation')
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--show_topk', type=bool, default=False, help='whether showing topk or not')
    parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')
    # parser.add_argument('--lr_decay', type=float, default=0.995, help="learning rate decay each round")
    # parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

    parser.add_argument('--rounds', type=int, default=200, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.05, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")  # 用于计算train_loader
    parser.add_argument('--dp_mechanism', type=str, default='Laplace', help='differential privacy mechanism')
    parser.add_argument('--dp_epsilon', type=float, default=0.1, help='differential privacy epsilon 0.01/0.02/0.03  0.1/1/2/5/10 ')
    parser.add_argument('--dp_delta', type=float, default=1e-5, help='differential privacy delta')
    parser.add_argument('--dp_clip', type=float, default=0.001, help='differential privacy clip  0.05/0.1/0.2/0.5  0.001/0.005/0.0005')
    '''

    # default settings for restaurant
    # 1.后缀output-文中给的参数是dim 8; n_hop 2; kge_w 0.01;l2_w 1e-7;lr 0.02;batch_size 1024;n_epoch 30;n_memory 32;neighbor_sample_size 4
    # 2.后缀不加output-代码原始参数：dim 8; n_hop 2; kge_w 0.01;l2_w 1e-7;lr 2e-2;batch_size 65536;n_epoch 30;n_memory 32;neighbor_sample_size 4
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='restaurant', help='which dataset to use')
    parser.add_argument('--dim', type=int, default=8, help='dimension of entity and relation embeddings')
    parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=30, help='the number of epochs')
    parser.add_argument('--n_memory', type=int, default=32, help='numbers of each ripple set')
    parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                        help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,
                        help='whether using outputs of all hops or just the last hop when making prediction')
    parser.add_argument('--n_heads', type=int, default=1, help='heads of self attention')
    parser.add_argument('--feed_f_dim', type=int, default=16, help='dim of feed forward network in transformer')
    parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
    parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
    '''

    args = parser.parse_args()
    return args


