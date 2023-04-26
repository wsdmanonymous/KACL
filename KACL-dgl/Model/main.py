from utility.helper import *
from utility.batch_test import *
from time import time

from GNN import myGAT
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

import multiprocessing
import os
import sys

cores = multiprocessing.cpu_count() // 2

def load_pretrained_data(args):
    pre_model = 'mf'
    pretrain_path = '%s../pretrain/%s/%s.npz' % (args.proj_path, args.dataset, pre_model)
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained bprmf model parameters.')
    except Exception:
        pretrain_data = None
    return pretrain_data


if __name__ == '__main__':
    torch.manual_seed(2023)
    np.random.seed(2023)
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    """
    *********************************************************
    Load Data from data_generator function.
    """
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_relations'] = data_generator.n_relations
    config['n_entities'] = data_generator.n_entities

    t0 = time()

    """
    *********************************************************
    Use the pretrained data to initialize the embeddings.
    """
    if args.pretrain in [-1, -2]:
        pretrain_data = load_pretrained_data(args)
    else:
        pretrain_data = None

    """
    *********************************************************
    Select one of the models.
    """
    weight_size = eval(args.layer_size)
    num_layers = len(weight_size) - 2
    heads = [args.heads] * num_layers + [1]
    print(config['n_users'], config['n_entities'], args.kge_size, config['n_relations'])
    
    model = myGAT(args, config['n_entities'], config['n_relations'] + 1, weight_size[-2], weight_size[-1], num_layers, heads, F.elu, 0.1, 0., 0.01, False, pretrain=pretrain_data).cuda()

    adjM = data_generator.lap_list

    print(len(adjM.nonzero()[0]))
    g = dgl.DGLGraph(adjM)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to('cuda')
    
    edge2type = {}
    for i,mat in enumerate(data_generator.kg_lap_list):
        for u,v in zip(*mat.nonzero()):
            edge2type[(u,v)] = i
    for i in range(data_generator.n_entities):
        edge2type[(i,i)] = len(data_generator.kg_lap_list)
    
    kg_adjM = sum(data_generator.kg_lap_list)
    kg = dgl.DGLGraph(kg_adjM)
    kg = dgl.remove_self_loop(kg)
    kg = dgl.add_self_loop(kg)
    e_feat = []
    for u, v in zip(*kg.edges()):
        u = u.item()
        v = v.item()
        if u == v:
            break
        e_feat.append(edge2type[(u,v)])
    for i in range(data_generator.n_entities):
        e_feat.append(edge2type[(i,i)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to('cuda')
    kg = kg.to('cuda')
    """
    *********************************************************
    Save the model parameters.
    """
    if args.save_flag == 1:
        weights_save_path = '{}weights/{}/{}/{}_{}.pt'.format(args.weights_path, args.dataset, args.model_type, num_layers, args.heads)
        ensureDir(weights_save_path)
        torch.save(model, weights_save_path)

    cur_best_pre_0 = 0.

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=args.kg_lr)
    optimizer3 = torch.optim.Adam(model.parameters(), lr=args.cl_lr)
    dropout_rate = args.drop_rate
    for epoch in range(args.epoch):
        t1 = time()
        sub_cf_adjM = data_generator._get_cf_adj_list(is_subgraph = True, dropout_rate = dropout_rate)
        sub_cf_lap = data_generator._get_lap_list(is_subgraph = True, subgraph_adj = sub_cf_adjM)

        sub_cf_g = dgl.DGLGraph(sub_cf_lap)
        sub_cf_g = dgl.add_self_loop(sub_cf_g)
        sub_cf_g = sub_cf_g.to('cuda')
        
        sub_kg_adjM, _ = data_generator._get_kg_adj_list(is_subgraph = True, dropout_rate = dropout_rate)
        sub_kg_lap = sum(data_generator._get_kg_lap_list(is_subgraph = True, subgraph_adj = sub_kg_adjM))
        sub_kg = dgl.DGLGraph(sub_kg_lap)
        sub_kg = dgl.remove_self_loop(sub_kg)
        sub_kg = dgl.add_self_loop(sub_kg)
        
        sub_kg = sub_kg.to('cuda')
        loss, base_loss, kge_loss, reg_loss, cl_loss = 0., 0., 0., 0., 0.
        cf_drop, kg_drop = 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        n_kg_batch = data_generator.n_triples // args.batch_size_kg + 1
        n_cl_batch = data_generator.n_items // args.batch_size_cl + 1
        """
        *********************************************************
        Alternative Training for KGAT:
        ... phase 1: to train the recommender.
        """
        for idx in range(n_batch):
            model.train()
            btime= time() 
            batch_data = data_generator.generate_train_batch()
            loss, cf_drop, kg_drop = model("cf", g, sub_cf_g, sub_kg, batch_data['users'], batch_data['pos_items'] +  data_generator.n_users, batch_data['neg_items'] + data_generator.n_users)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
        for idx in range(n_kg_batch):
            model.train()
            batch_data = data_generator.generate_train_kg_batch()
            kge_loss = model("kg", sub_kg, batch_data['heads'], batch_data['relations'], batch_data['pos_tails'], batch_data['neg_tails'])

            optimizer2.zero_grad()
            kge_loss.backward()
            optimizer2.step()
        
        for idx in range(n_cl_batch):
            model.train()
            batch_data = data_generator.generate_train_cl_batch()
            cl_loss = model("cl", sub_cf_g, sub_kg, batch_data['items'])
            
            optimizer3.zero_grad()
            cl_loss.backward()
            optimizer3.step()

        del sub_cf_g, sub_kg
        show_step = 10
        if (epoch + 1) % show_step != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f + %.5f + %.5f] drop==[%.2f + %.2f]' % (
                    epoch, time() - t1, float(loss), float(kge_loss), float(cl_loss), float(cf_drop), float(kg_drop))
                print(perf_str)
            continue
        

        """
        *********************************************************
        Test.
        """
        
        t2 = time()
        users_to_test = list(data_generator.test_user_dict.keys())

        ret = test(g, kg, model, users_to_test)
        """
        *********************************************************
        Performance logging.
        """
        t3 = time()

        loss_loger.append(float(loss))
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, float(loss), float(kge_loss), ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=10)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break
   
        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            # save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            torch.save(model, weights_save_path)
            print('save the weights in path: ', weights_save_path)
            print('saving prediction')
            #save_file(g, e_feat, model, users_to_test)
            print('saved')
            # print(test_saved_file(users_to_test))

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, args.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write('embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s, use_att=%s, use_kge=%s, pretrain=%d\n\t%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs, args.adj_type, args.use_att, args.use_kge, args.pretrain, final_perf))
    f.close()
