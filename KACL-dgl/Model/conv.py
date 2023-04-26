"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

class DropLearner(nn.Module):
    def __init__(self, node_dim, edge_dim = None, mlp_edge_model_dim = 64):
        super(DropLearner, self).__init__()
        
        self.mlp_src = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.mlp_dst = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.mlp_con = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        
        self.concat = False
        
        if edge_dim is not None:
            self.mlp_edge = nn.Sequential(
                nn.Linear(edge_dim, mlp_edge_model_dim),
                nn.ReLU(),
                nn.Linear(mlp_edge_model_dim, 1)
            )
        else:
            self.mlp_edge = None
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    
    def get_weight(self, head_emb, tail_emb, temperature = 0.5, relation_emb = None, edge_type = None):
        if self.concat:
            weight = self.mlp_con(head_emb + tail_emb)
            w_src = self.mlp_src(head_emb)
            w_dst = self.mlp_dst(tail_emb)
            weight += w_src + w_dst
        else:
            w_src = self.mlp_src(head_emb)
            w_dst = self.mlp_dst(tail_emb)
            weight = w_src + w_dst
        if relation_emb is not None and self.mlp_edge is not None:
            e_weight = self.mlp_edge(relation_emb)
            weight += e_weight
        weight = weight.squeeze()
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * th.rand(weight.size()) + (1 - bias)
        gate_inputs = th.log(eps) - th.log(1 - eps)
        gate_inputs = gate_inputs.to(head_emb.device)
        gate_inputs = (gate_inputs + weight) / temperature
        aug_edge_weight = th.sigmoid(gate_inputs).squeeze()
        edge_drop_out_prob = 1 - aug_edge_weight
        reg = edge_drop_out_prob.mean()
        #print(aug_edge_weight.size())
        return reg.detach(), aug_edge_weight.detach()
    
    def forward(self, node_emb, graph, temperature = 0.5, relation_emb = None, edge_type = None):
        if self.concat:
            w_con = node_emb
            graph.srcdata.update({'in': w_con})
            graph.apply_edges(fn.u_add_v('in', 'in', 'con'))
            n_weight = graph.edata.pop('con')
            weight = self.mlp_con(n_weight)
            w_src = self.mlp_src(node_emb)
            w_dst = self.mlp_dst(node_emb)
            graph.srcdata.update({'inl': w_src})
            graph.dstdata.update({'inr': w_dst})
            graph.apply_edges(fn.u_add_v('inl', 'inr', 'ine'))
            weight += graph.edata.pop('ine')
            #print(weight.size())
        else:
            w_src = self.mlp_src(node_emb)
            w_dst = self.mlp_dst(node_emb)
            graph.srcdata.update({'inl': w_src})
            graph.dstdata.update({'inr': w_dst})
            graph.apply_edges(fn.u_add_v('inl', 'inr', 'ine'))
            n_weight = graph.edata.pop('ine')
            weight = n_weight
        if relation_emb is not None and self.mlp_edge is not None:
            w_edge = self.mlp_edge(relation_emb)
            graph.edata.update({'ee': w_edge})
            e_weight = graph.edata.pop('ee')
            weight += e_weight
        weight = weight.squeeze()
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * th.rand(weight.size()) + (1 - bias)
        gate_inputs = th.log(eps) - th.log(1 - eps)
        gate_inputs = gate_inputs.to(node_emb.device)
        gate_inputs = (gate_inputs + weight) / temperature
        aug_edge_weight = th.sigmoid(gate_inputs).squeeze()
        edge_drop_out_prob = 1 - aug_edge_weight
        reg = edge_drop_out_prob.mean()
        aug_edge_weight = aug_edge_weight.unsqueeze(-1).unsqueeze(-1)
        #print(aug_edge_weight.size())
        return reg, aug_edge_weight
        


# pylint: enable=W0235
class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0., attn_drop=0.,
                 negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=False, alpha=0.):
        super(myGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, res_attn=None, edge_weight = None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if edge_weight is not None:
                graph.edata['a'] = graph.edata['a'] * edge_weight
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()

