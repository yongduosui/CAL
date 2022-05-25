import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout
import pdb
import dgl


class CausalReadout_complex(nn.Module):

    def __init__(self, input_dim, output_dim, args, readout=None): #L=nb_hidden_layers
        super().__init__()

        self.model = args.model
        self.edge_att_mlp = nn.Linear(input_dim * 2, 2)
        self.node_att_mlp = nn.Linear(input_dim, 2)

        self.context_convs = GCNLayer(input_dim, input_dim, F.relu, dropout=0, batch_norm=True)
        self.objects_convs = GCNLayer(input_dim, input_dim, F.relu, dropout=0, batch_norm=True)
        
        if "GIN" in self.model:
            self.context_mlps = nn.Linear(input_dim, output_dim)
            self.objects_mlps = nn.Linear(input_dim, output_dim) 
            self.concats_mlps = nn.Linear(input_dim, output_dim)
        else:
            self.context_mlps = MLPReadout(input_dim, output_dim)
            self.objects_mlps = MLPReadout(input_dim, output_dim) 
            self.concats_mlps = MLPReadout(input_dim, output_dim)
        self.random_or_avg = args.random_or_avg
        self.readout = readout
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        self.without_random = args.without_random

    def forward(self, g, h, eval_random=True):
        
        row, col = g.edges()[0], g.edges()[1]
        edge_rep = torch.cat([h[row], h[col]], dim=-1)
        # edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        # node_att = F.softmax(self.node_att_mlp(h), dim=-1)
        if self.without_node_attention:
            node_att = 0.5 * torch.ones(h.shape[0], 2).cuda()
        else:
            node_att = F.softmax(self.node_att_mlp(h), dim=-1)
        
        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2).cuda()
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)

        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]
        node_weight_c = node_att[:, 0]
        node_weight_o = node_att[:, 1]

        xc = node_weight_c.view(-1, 1) * h
        xo = node_weight_o.view(-1, 1) * h

        xc = self.context_convs(g, xc, data_mask=edge_weight_c)
        xo = self.objects_convs(g, xo, data_mask=edge_weight_o)

        if self.readout is not None:
            hg_xc = self.readout(g, xc)
            hg_xo = self.readout(g, xo)
        else:
            g.ndata['xc'] = xc
            g.ndata['xo'] = xo
            hg_xc = dgl.mean_nodes(g, 'xc')
            hg_xo = dgl.mean_nodes(g, 'xo')
        
        if self.random_or_avg == "random":
            hg_xco = self.random_out(hg_xc, hg_xo, eval_random)
        elif self.random_or_avg == "avg":
            hg_xco = self.average_out(hg_xc, hg_xo)    
        else:
            assert False

        c_out = self.context_mlps(hg_xc)
        o_out = self.objects_mlps(hg_xo)
        co_out = self.concats_mlps(hg_xco)
        return c_out, o_out, co_out

    def random_out(self, xc, xo, eval_random):

        num = xo.shape[0]
        l = [i for i in range(num)]
        
        if not self.without_random:
            if eval_random:
                random.shuffle(l)
        random_idx = torch.tensor(l)
        xco = xo + xc[random_idx]
        return xco

    def average_out(self, xc, xo):

        rol, col = xc.shape
        avg_context = xc.mean(dim=0).expand(rol, col)
        xco = avg_context + xo
        return xco


