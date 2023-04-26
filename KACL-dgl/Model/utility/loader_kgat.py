import numpy as np
from utility.load_data import Data

import scipy.sparse as sp
import random as rd
import collections
from time import time

class KGAT_loader(Data):
    def __init__(self, args, path):
        super().__init__(args, path)
        self.all_kg_dict = self._get_all_kg_dict()
        # generate the sparse adjacency matrices for user-item interaction.
        self.adj_list= self._get_cf_adj_list()
        self.kg_adj_list, self.adj_r_list = self._get_kg_adj_list()

        # generate the sparse laplacian matrices.
        self.lap_list = self._get_lap_list()
        self.kg_lap_list = self._get_kg_lap_list()
        
        # generate the triples dictionary, key is 'head', value is '(tail, relation)'.
    
    def _get_cf_adj_list(self, is_subgraph = False, dropout_rate = None):
        def _np_mat2sp_adj(np_mat, row_pre, col_pre):
            n_all = self.n_users + self.n_items
            # single-direction
            a_rows = np_mat[:, 0] + row_pre
            a_cols = np_mat[:, 1] + col_pre
            if is_subgraph is True:
                subgraph_idx = np.arange(len(a_rows))
                subgraph_id = np.random.choice(subgraph_idx, size = int(dropout_rate * len(a_rows)), replace = False)
                a_rows = a_rows[subgraph_id]
                a_cols = a_cols[subgraph_id]
                
            vals = [1.] * len(a_rows) * 2
            rows = np.concatenate((a_rows, a_cols))
            cols = np.concatenate((a_cols, a_rows))
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(n_all, n_all))
            return adj
        R = _np_mat2sp_adj(self.train_data, row_pre=0, col_pre=self.n_users)
        return R

    def _get_kg_adj_list(self, is_subgraph = False, dropout_rate = None):
        adj_mat_list = []
        adj_r_list = []
        def _np_mat2sp_adj(np_mat):
            n_all = self.n_entities
            # single-direction
            a_rows = np_mat[:, 0]
            a_cols = np_mat[:, 1]
            if is_subgraph is True:
                subgraph_idx = np.arange(len(a_rows))
                subgraph_id = np.random.choice(subgraph_idx, size = int(dropout_rate * len(a_rows)), replace = False)
                #print(subgraph_id[:10])
                a_rows = a_rows[subgraph_id]
                a_cols = a_cols[subgraph_id]
            a_vals = [1.] * len(a_rows)
            
            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))

            return a_adj, b_adj
        
        for r_id in self.relation_dict.keys():
            #print(r_id)
            K, K_inv = _np_mat2sp_adj(np.array(self.relation_dict[r_id]))
            adj_mat_list.append(K)
            adj_r_list.append(r_id)

            adj_mat_list.append(K_inv)
            adj_r_list.append(r_id + self.n_relations)
        self.n_relations = self.n_relations * 2
        #print(adj_r_list)
        return adj_mat_list, adj_r_list

    def _bi_norm_lap(self, adj):
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(self, adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()
    
    def _get_kg_lap_list(self, is_subgraph = False, subgraph_adj = None):
        if is_subgraph is True:
            adj_list = subgraph_adj
        else:
            adj_list = self.kg_adj_list
        if self.args.adj_type == 'bi':
            lap_list = [self._bi_norm_lap(adj) for adj in adj_list]
        else:
            lap_list = [self._si_norm_lap(adj) for adj in adj_list]
        return lap_list
        
    def _get_lap_list(self, is_subgraph = False, subgraph_adj = None):
        if is_subgraph is True:
            adj = subgraph_adj
        else:
            adj = self.adj_list
        if self.args.adj_type == 'bi':
            lap_list = self._bi_norm_lap(adj)
        else:
            lap_list = self._si_norm_lap(adj)
        return lap_list

    def _get_all_kg_dict(self):
        all_kg_dict = collections.defaultdict(list)
        
        for relation in self.relation_dict.keys():
            for head, tail in self.relation_dict[relation]:
                all_kg_dict[head].append((tail, relation))
                all_kg_dict[tail].append((head, relation + self.n_relations))
        return all_kg_dict

    def _generate_train_cf_batch(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users_list = list(self.exist_users)
            users = [rd.choice(users_list) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_user_dict[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch
        
        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_i_id = np.random.randint(low=0, high=self.n_items,size=1)[0]

                if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                    neg_items.append(neg_i_id)
            return neg_items
        
        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
        return users, pos_items, neg_items


    def _generate_train_cl_batch(self):
        if self.batch_size_cl <= len(self.exist_items):
            items = rd.sample(self.exist_items, self.batch_size_cl)
        else:
            items_list = list(self.exist_items)
            items = [rd.choice(items_list) for _ in range(self.batch_size_cl)]
        return items
    
    def _generate_train_kg_batch(self):
        exist_heads = self.all_kg_dict.keys()

        if self.batch_size_kg <= len(exist_heads):
            heads = rd.sample(exist_heads, self.batch_size_kg)
        else:
            heads = [rd.choice(exist_heads) for _ in range(self.batch_size_kg)]

        def sample_pos_triples_for_h(h, num):
            pos_triples = self.all_kg_dict[h]
            n_pos_triples = len(pos_triples)

            pos_rs, pos_ts = [], []
            while True:
                if len(pos_rs) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
                t = pos_triples[pos_id][0]
                r = pos_triples[pos_id][1]

                if r not in pos_rs and t not in pos_ts:
                    pos_rs.append(r)
                    pos_ts.append(t)
            return pos_rs, pos_ts

        def sample_neg_triples_for_h(h, r, num):
            neg_ts = []
            while True:
                if len(neg_ts) == num: break

                t = np.random.randint(low=0, high=self.n_entities, size=1)[0]
                if (t, r) not in self.all_kg_dict[h] and t not in neg_ts:
                    neg_ts.append(t)
            return neg_ts
        
        pos_r_batch, pos_t_batch, neg_t_batch = [], [], []

        for h in heads:
            pos_rs, pos_ts = sample_pos_triples_for_h(h, 1)
            pos_r_batch += pos_rs
            pos_t_batch += pos_ts

            neg_ts = sample_neg_triples_for_h(h, pos_rs[0], 1)
            neg_t_batch += neg_ts

        return heads, pos_r_batch, pos_t_batch, neg_t_batch

    def generate_train_batch(self):
        
        users, pos_items, neg_items = self._generate_train_cf_batch()

        batch_data = {}
        batch_data['users'] = users
        batch_data['pos_items'] = pos_items
        batch_data['neg_items'] = neg_items
        return batch_data
        

    def generate_train_kg_batch(self):
        heads, relations, pos_tails, neg_tails = self._generate_train_kg_batch()

        batch_data = {}

        batch_data['heads'] = heads
        batch_data['relations'] = relations
        batch_data['pos_tails'] = pos_tails
        batch_data['neg_tails'] = neg_tails
        return batch_data
    
    def generate_train_cl_batch(self):
        items = self._generate_train_cl_batch()
        batch_data = {}
        batch_data['items'] = items
        return batch_data


