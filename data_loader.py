# -*- coding:utf-8 -*-
import collections
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import random


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data, user_history_dict = load_rating(args)
    n_entity, n_relation, kg ,adj_entity, adj_relation= load_kg(args)
    ripple_set = get_ripple_set(args, kg, user_history_dict)
    return n_user, n_item, train_data, eval_data, test_data, n_entity, n_relation, ripple_set, adj_entity, adj_relation


def load_rating(args):
    # 读取评分文件
    print('reading rating file ...')
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data, user_history_dict = dataset_split(rating_np)
    return n_user, n_item, train_data, eval_data, test_data, user_history_dict   # music 可以执行


def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    # print(len(train_indices), len(eval_indices), len(test_indices))

    # 遍历训练数据，只保留有正面评价的用户
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]
    # print(len(train_indices), len(eval_indices), len(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data, user_history_dict  # music 可以执行


def load_kg(args):
    # 读取知识图谱文件，将其转换为向量npy文件
    print('reading KG file ...')
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)
    adj_entity, adj_relation = construct_adj(args, kg, n_entity)

    return n_entity, n_relation, kg, adj_entity, adj_relation


def construct_kg(kg_np):
    # 构建知识图谱
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


def get_ripple_set(args, kg, user_history_dict):
    # 构建RippleNet集合
    print('constructing ripple set ...')

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for user in user_history_dict:
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])

            # 如果给定用户的当前ripple集是空的，我们只需将最后一跳的波纹集复制到这里。
            # 这不会出现在h=0的情况下，因为物品只有出现在KG中的项目才被选中。
            # 这只出现在Book Crossing数据集的154个用户身上（因为BX数据集和KG都是稀疏的）。
            if len(memories_h) == 0:
                ripple_set[user].append(ripple_set[user][-1])
            else:
                # 为每个用户取样一个固定大小的1跳内存
                replace = len(memories_h) < args.n_memory
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[user].append((memories_h, memories_r, memories_t))

    return ripple_set

def construct_adj(args, kg, n_entity):  #创建邻接矩阵
    print('constructing adjacency matrix ...')
    # adj_entity的每一行都存储了一个给定实体的采样邻居实体。
    # adj_relation的每一行都存储了相应的采样邻居关系。
    adj_entity = np.zeros([n_entity, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([n_entity, args.neighbor_sample_size], dtype=np.int64)

    for entity in range(n_entity):
        if entity in kg:
            neighbors = kg[entity]
            n_neighbors = len(neighbors)

            if n_neighbors >= args.neighbor_sample_size:
                sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
            else:
                sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
            adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
            adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation



class DataLoader:
    '''
    Data Loader class which makes dataset for training / knowledge graph dictionary
    '''
    def __init__(self, data):
        self.cfg = {
            'movie-1m': {
                'item2id_path': '../data/movie-1m/item_index2entity_id.txt',
                'kg_path': '../data/movie-1m/kg.txt',
                'rating_path': '../data/movie-1m/ratings.dat',
                'rating_sep': '::',
                'threshold': 4.0
            },
            'movie-20m': {
                'item2id_path': '../data/movie-20m/item_index2entity_id.txt',
                'kg_path': '../data/movie-20m/kg.txt',
                'rating_path': '../data/movie-20m/ratings.csv',
                'rating_sep': ',',
                'threshold': 4.0
            },
            'music': {
                'item2id_path': '../data/music/item_index2entity_id.txt',
                'kg_path': '../data/music/kg.txt',
                'rating_path': '../data/music/user_artists.dat',
                'rating_sep': '\t',
                'threshold': 0.0
            },
            'book': {
                'item2id_path': '../data/book/item_index2entity_id.txt',
                'kg_path': '../data/book/kg.txt',
                'rating_path': '../data/book/BX-Book-Ratings.csv',
                'rating_sep': ';',
                'threshold': 0.0
            }
        }
        self.data = data
        
        df_item2id = pd.read_csv(self.cfg[data]['item2id_path'], sep='\t', header=None, names=['item','id'])
        df_kg = pd.read_csv(self.cfg[data]['kg_path'], sep='\t', header=None, names=['head','relation','tail'])
        df_rating = pd.read_csv(self.cfg[data]['rating_path'], sep=self.cfg[data]['rating_sep'], names=['userID', 'itemID', 'rating'], skiprows=1)
        
        # df_rating['itemID'] and df_item2id['item'] both represents old entity ID
        df_rating = df_rating[df_rating['itemID'].isin(df_item2id['item'])]
        df_rating.reset_index(inplace=True, drop=True)
        
        self.df_item2id = df_item2id
        self.df_kg = df_kg
        self.df_rating = df_rating
        
        self.user_encoder = LabelEncoder()
        self.entity_encoder = LabelEncoder()
        self.relation_encoder = LabelEncoder()

        self._encoding()
        
    def _encoding(self):
        '''
        Fit each label encoder and encode knowledge graph
        '''
        self.user_encoder.fit(self.df_rating['userID'])
        # df_item2id['id'] and df_kg[['head', 'tail']] represents new entity ID
        self.entity_encoder.fit(pd.concat([self.df_item2id['id'], self.df_kg['head'], self.df_kg['tail']]))
        self.relation_encoder.fit(self.df_kg['relation'])
        
        # encode df_kg
        self.df_kg['head'] = self.entity_encoder.transform(self.df_kg['head'])
        self.df_kg['tail'] = self.entity_encoder.transform(self.df_kg['tail'])
        self.df_kg['relation'] = self.relation_encoder.transform(self.df_kg['relation'])
        
    def _build_dataset(self):
        '''
        Build dataset for training (rating data)
        It contains negative sampling process
        '''
        print('Build dataset dataframe ...', end=' ')
        # df_rating update
        df_dataset = pd.DataFrame()
        df_dataset['userID'] = self.user_encoder.transform(self.df_rating['userID'])
        
        # update to new id
        item2id_dict = dict(zip(self.df_item2id['item'], self.df_item2id['id']))
        self.df_rating['itemID'] = self.df_rating['itemID'].apply(lambda x: item2id_dict[x])
        df_dataset['itemID'] = self.entity_encoder.transform(self.df_rating['itemID'])
        df_dataset['label'] = self.df_rating['rating'].apply(lambda x: 0 if x < self.cfg[self.data]['threshold'] else 1)
        
        # negative sampling
        df_dataset = df_dataset[df_dataset['label']==1]
        # df_dataset requires columns to have new entity ID
        full_item_set = set(range(len(self.entity_encoder.classes_)))
        user_list = []
        item_list = []
        label_list = []
        for user, group in df_dataset.groupby(['userID']):
            item_set = set(group['itemID'])
            negative_set = full_item_set - item_set
            negative_sampled = random.sample(negative_set, len(item_set))
            user_list.extend([user] * len(negative_sampled))
            item_list.extend(negative_sampled)
            label_list.extend([0] * len(negative_sampled))
        negative = pd.DataFrame({'userID': user_list, 'itemID': item_list, 'label': label_list})
        df_dataset = pd.concat([df_dataset, negative])
        
        df_dataset = df_dataset.sample(frac=1, replace=False, random_state=999)
        df_dataset.reset_index(inplace=True, drop=True)
        print('Done')
        return df_dataset
    
    def load_dataset(self):
        return self._build_dataset()