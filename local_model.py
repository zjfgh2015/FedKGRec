# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score,f1_score
from aggregator import Aggregator
import random
import math


class KGNN(nn.Module): # RippleNet融合GCN 多了adj_entity,adj_relation两个参数
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation): # 初始化使用 args n_entity n_relation
        super(KGNN, self).__init__()

        self._parse_args(args, n_user, n_entity, n_relation)  #RippleNet
        self.user_emb = nn.Embedding(self.n_user, self.dim)   #KGCN
        self.entity_emb = nn.Embedding(self.n_entity, self.dim) #RippleNet
        self.relation_emb = nn.Embedding(self.n_relation, self.dim * self.dim) #RippleNet
        self.relation_emb_GCN = nn.Embedding(self.n_relation, self.dim) #KGCN
        self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False) #RippleNet
        self.criterion = nn.BCELoss() #RippleNet
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.n_head, dim_feedforward=self.feed_f_dim)
        self.transformerEncoder = nn.TransformerEncoder(num_layers=1, encoder_layer=self.transformerEncoderLayer)
        self.pooling = nn.AvgPool2d
        self.linear = nn.Linear(in_features=2 * self.dim, out_features=self.dim)
        self.adj_entity = adj_entity #KGCN
        self.adj_relation = adj_relation #KGCN
        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator) #KGCN
        self._init_weight()


    def _parse_args(self, args, n_user, n_entity, n_relation):
        # RippleNet融合KGCN的参数列表
        self.n_user = n_user
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops
        self.n_head = args.n_heads  # KGCN
        self.feed_f_dim = args.feed_f_dim # KGCN
        self.n_iter = args.n_iter # KGCN
        self.batch_size = args.batch_size # KGCN
        self.n_neighbor = args.neighbor_sample_size # KGCN

    def _init_weight(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward( # RippleNet
        self,
        users: torch.LongTensor,
        items: torch.LongTensor,   # 一个 batch 的items
        labels: torch.LongTensor,   # 一个 batch 的labels
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        # batch_size = items.size(0)  # KGCN
        batch_size = users.size(0)  # KGCN
        if batch_size != self.batch_size:
            self.batch_size = batch_size

        # [batch size, dim]
        item_embeddings_ripple = self.entity_emb(items)   # RippleNet的item_embeddings
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_list.append(
                self.relation_emb(memories_r[i]).view(
                    -1, self.n_memory, self.dim, self.dim
                )
            )
            # [batch size, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[i]))

        o_list, item_embeddings_ripple = self._key_addressing(
            h_emb_list, r_emb_list, t_emb_list, item_embeddings_ripple
        )

        h_rep = self._history_extracting(h_emb_list)
        # user_embeddings = self._history_extracting(h_emb_list) #KGCN的user_embeddings
        user_embedding_ripple = self.user_emb(users)  #原始RippleNet没有user表示

        # entities: list   ele1 [batch size 1]    ele2 [batch size  8]   ele3 [batch size  64 ]  ele4 [batch size  512(64 * 8)]
        entities, relations = self._get_neighbors(items)   #KGCN

        #[batch dim ]
        # item_embeddings = self._aggregate(h_rep, entities, relations)  #KGCN的item_embeddings
        item_embeddings = self._aggregate(user_embedding_ripple, entities, relations)  #KGCN的item_embeddings

        #o_list.append(u_history_embedding)
        # scores = self.predict(item_embeddings, o_list) #原始RippleNet
        scores = self.predict(item_embeddings_ripple, user_embedding_ripple, item_embeddings, h_rep) #融合GCN之后
        # scores = self.predict(item_embeddings_ripple,item_embeddings,h_rep) #融合GCN之后

        return_dict = self._compute_loss(  #RippleNet
            scores, labels, h_emb_list, t_emb_list, r_emb_list
        )
        return_dict["scores"] = scores
        return return_dict

    def _compute_loss(self, scores, labels, h_emb_list, t_emb_list, r_emb_list):
        # 原始RippleNet的计算损失函数
        base_loss = self.criterion(scores, labels.float())

        kge_loss = 0
        for hop in range(self.n_hop):
            # [batch size, n_memory, 1, dim]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=2)
            # [batch size, n_memory, dim, 1]
            t_expanded = torch.unsqueeze(t_emb_list[hop], dim=3)
            # [batch size, n_memory, dim, dim]
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
            )
            kge_loss += torch.sigmoid(hRt).mean()
        kge_loss = -self.kge_weight * kge_loss

        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss += (h_emb_list[hop] * h_emb_list[hop]).sum()
            l2_loss += (t_emb_list[hop] * t_emb_list[hop]).sum()
            l2_loss += (r_emb_list[hop] * r_emb_list[hop]).sum()
        l2_loss = self.l2_weight * l2_loss

        loss = base_loss + kge_loss + l2_loss
        return dict(base_loss=base_loss, kge_loss=kge_loss, l2_loss=l2_loss, loss=loss)

    def _key_addressing(self, h_emb_list, r_emb_list, t_emb_list, item_embeddings):
        #原始RippleNet
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]  h扩充一维 [1024 32 16 1]
            h_expanded = torch.unsqueeze(h_emb_list[hop], dim=3)

            # [batch_size, n_memory, dim]  Rh   [1024 32 16 1]  =   [1024 32 16 16] mat [1024 32 16 1 ]
            Rh = torch.squeeze(torch.matmul(r_emb_list[hop], h_expanded))

            # [batch_size, dim, 1]   [1024 16 1 ]
            v = torch.unsqueeze(item_embeddings, dim=2)

            # [batch_size, n_memory] [1024 32]
            probs = torch.squeeze(torch.matmul(Rh, v))

            # [batch_size, n_memory]
            probs_normalized = F.softmax(probs, dim=1)

            # [batch_size, n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)

            # [batch_size, dim]
            o = (t_emb_list[hop] * probs_expanded).sum(dim=1)

            item_embeddings = self._update_item_embedding(item_embeddings, o)
            o_list.append(o)
        return o_list, item_embeddings

    def _update_item_embedding(self, item_embeddings, o):
        #原始RippleNet的更新物品向量函数
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = self.transform_matrix(o)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = self.transform_matrix(item_embeddings + o)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    def _history_extracting(self,h_emb_list):
        #新添加的历史记录提取函数
        #[batch_size n_memeres  dim]
        history =h_emb_list[0]
        #[1024 32 16 ]
        history = self.transformerEncoder(history)
        pool = self.pooling(kernel_size=(self.n_memory,1),stride=(self.n_memory,1))
        h_rep = pool(history).squeeze()
        return h_rep

    def predict(self, item_embedding_ripple, user_embeddings_ripple, item_embeddings, u_rep ):
        #原始RippleNet predict(self, item_embeddings, o_list) 融合KGCN
        item_rep = torch.cat((item_embedding_ripple,item_embeddings),dim=1)
        item_repsentation = self.linear(item_rep)
        user_rep = torch.cat((user_embeddings_ripple,u_rep),dim=1)
        user_representation = self.linear(user_rep)
        scores = (item_repsentation * user_representation).sum(dim=1)
        # [batch_size]
        #scores = (item_embeddings * y).sum(dim=1)
        return torch.sigmoid(scores)

    def evaluate(self, users, items, labels, memories_h, memories_r, memories_t):
        #原始RippleNet的评估函数
        return_dict = self.forward(users, items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()
        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        f1 = f1_score(y_true=labels, y_pred=predictions)
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc, f1

    def _get_neighbors(self, items): #KGCN的函数
        # KGCN中的获取领域节点的函数
        '''
        v 是项目的批量大小的索引
        v: [batch_size, 1]
        '''
        #seeds [batch_size  1 ]
        seeds =torch.unsqueeze(items,dim=1)
        #entities: list[1]    #[batch_size  1]
        entities = [seeds]

        relations = []

        for h in range(self.n_iter):
            n_e = self.adj_entity[entities[h].cpu()]
            n_r = self.adj_relation[entities[h].cpu()]
            neighbor_entities = torch.LongTensor(n_e).view((self.batch_size, -1)).cuda()
            neighbor_relations = torch.LongTensor(n_r).view((self.batch_size, -1)).cuda()
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        return entities, relations

    def _aggregate(self, user_embeddings, entities, relations):
        # KGCN中的聚合的函数
        '''
        通过聚合邻居向量进行项目嵌入
        user_embedding[batch size   dim ]
        '''
        # entity_vectors list:3  0: [batch size 1 dim], 1 [batch size  8  dim ],2 [batch 64 dim ], 3  [batch 64*8   dim]
        entity_vectors = [self.entity_emb(entity) for entity in entities]
        # relation_vectors list2  0: [batch 8 dim] , 1 :[batch 64  dim]
        relation_vectors = [self.relation_emb_GCN(relation) for relation in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].view((self.batch_size, self.dim))

# if __name__ == '__main__':
#     model = RippleNet()
#     print(model)