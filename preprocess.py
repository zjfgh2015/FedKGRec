# -*- coding:utf-8 -*-
import argparse
import numpy as np
import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)
RATING_FILE_NAME = dict({'movie-1m': 'ratings.dat', 'movie-20m': 'ratings.csv', 'book': 'BX-Book-Ratings.csv', 'music': 'user_artists.dat'})
SEP = dict({'movie-1m': '::', 'movie-20m': ',', 'book': ';', 'music': '\t'}) #设置分隔符
THRESHOLD = dict({'movie-1m': 4, 'movie-20m': 4, 'book': 0, 'music': 0})


def read_item_index_to_entity_id_file(DATASET):
    # 读取文件方法 物品的索引——>实体的id
    file = '../data/' + DATASET + '/item_index2entity_id.txt'
    # print('reading item index to entity id file: ' + file + ' ...')
    logging.info("reading item index to entity id file: %s", file)
    item_index_old2new = dict()
    entity_id2index = dict()
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]  # (strip 移除字符串首尾 字符)
        satori_id = line.strip().split('\t')[1]  # Satori实体ID
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1
    return item_index_old2new, entity_id2index

def convert_rating(DATASET, item_index_old2new, entity_id2index):
    #  转换rating文件 ratings.dat/csv——>ratings_final.txt
    file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]
    logging.info("reading rating file: %s", file)

    # print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[DATASET])

        # 删除BX数据集的前缀和后缀引号
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # 该项目不在最终项目集中
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])
        if rating >= THRESHOLD[DATASET]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    # print('converting rating file ...')
    write_file = '../data/' + DATASET + '/ratings_final.txt'
    logging.info("converting rating file to: %s", write_file)
    writer = open(write_file, 'w', encoding='utf-8')
    writer_idx = 0
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer_idx += 1
            writer.write('%d\t%d\t1\n' % (user_index, item))
        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer_idx += 1
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    # print('number of users: %d' % user_cnt)
    # print('number of items: %d' % len(item_set))
    logging.info("number of users: %d", user_cnt)
    logging.info("number of items: %d", len(item_set))
    logging.info("number of interactions: %d", writer_idx)


def convert_kg(DATASET, entity_id2index):
    # 转换kg文件 kg.txt——>kg_final.txt
    # print('converting kg file ...')
    # entity_cnt = len(entity_id2index)
    # relation_cnt = 0
    file = '../data/' + DATASET + '/' + 'kg.txt'
    logging.info("reading kg file: %s", file)
    write_file = '../data/' + DATASET + '/' + 'kg_final.txt'
    logging.info("converting kg file to: %s", write_file)

    entity_cnt = len(entity_id2index)
    relation_id2index = dict()
    relation_cnt = 0

    writer = open(write_file, 'w', encoding='utf-8')
    writer_idx = 0
    files = []
    if DATASET == 'movie-1m':
        files.append(open('../data/' + DATASET + '/kg_part1.txt', encoding='utf-8'))
        files.append(open('../data/' + DATASET + '/kg_part2.txt', encoding='utf-8'))
    else:
        files.append(open('../data/' + DATASET + '/kg.txt', encoding='utf-8'))

    for file in files:
        for line in file:
            array = line.strip().split('\t')
            head_old = array[0]
            relation_old = array[1]
            tail_old = array[2]

            if head_old not in entity_id2index:
                entity_id2index[head_old] = entity_cnt
                entity_cnt += 1
            head = entity_id2index[head_old]

            if tail_old not in entity_id2index:
                entity_id2index[tail_old] = entity_cnt
                entity_cnt += 1
            tail = entity_id2index[tail_old]

            if relation_old not in relation_id2index:
                relation_id2index[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation_id2index[relation_old]

            writer.write('%d\t%d\t%d\n' % (head, relation, tail))
            writer_idx += 1

    writer.close()
    # print('number of entities (containing items): %d' % entity_cnt)
    # print('number of relations: %d' % relation_cnt)
    logging.info("number of entities (containing items): %d", entity_cnt)
    logging.info("number of relations: %d", relation_cnt)
    logging.info("number of triples: %d", writer_idx)
    return entity_id2index, relation_id2index

if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='movie-20m', help='which dataset to preprocess')
#     args = parser.parse_args()
    args =parser.parse_known_args()[0]
    DATASET = args.dataset

    # entity_id2index = dict() # entity_id2index 是一个字典 key是实体id  value是索引id
    # relation_id2index = dict()
    # item_index_old2new = dict()  # item_index_old2new是将不规则的item id 重新排列 (对齐)
    #
    # read_item_index_to_entity_id_file()
    # convert_rating()
    # convert_kg()

    item_index_old2new, entity_id2index = read_item_index_to_entity_id_file(DATASET) # item_index_old2new 是将不规则的item id 重新排列 (对齐)
    convert_rating(DATASET, item_index_old2new, entity_id2index)
    entity_id2index, relation_id2index = convert_kg(DATASET, entity_id2index) # entity_id2index 是一个字典 key是实体id  value是索引id

    logging.info("data %s preprocess: done.", DATASET)
