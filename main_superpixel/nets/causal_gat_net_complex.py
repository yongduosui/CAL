import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from layers.gat_layer import GATLayer
from layers.mlp_readout_layer import MLPReadout
from layers.causal_readout_layer import CausalReadout_complex
import pdb

class CausalGAT_complex(nn.Module):
    def __init__(self, net_params, args):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        self.n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.embedding_h = nn.Linear(in_dim, hidden_dim * num_heads)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList([GATLayer(hidden_dim * num_heads, 
                                              hidden_dim, 
                                              num_heads,
                                              dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual))
        self.Causal_layer = CausalReadout_complex(out_dim, self.n_classes, args) 

    def forward(self, g, h, e, eval_random=True):

        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(g, h)
        xc, xo, xco = self.Causal_layer(g, h, eval_random)
        return xc, xo, xco

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss

    def kl_loss(self, pred, label):
        xco_logis = F.log_softmax(pred, dim=-1)
        loss = F.kl_div(xco_logis, label, reduction='batchmean')
        return loss