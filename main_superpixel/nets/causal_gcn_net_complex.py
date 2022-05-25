import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout
from layers.causal_readout_layer import CausalReadout_complex
import pdb

def print_model_info(net_params):
    print("model information")
    print("-" *  100)
    for k, v in net_params.items():
        print("{} -> {}".format(k, v))
    print("-" *  100)


class CausalGCN_complex(nn.Module):
    def __init__(self, net_params, args):
        super().__init__()
        
        self.args = args
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        self.n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu, dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual))
        self.Causal_layer = CausalReadout_complex(out_dim, self.n_classes, args)        
        print_model_info(net_params)
        
    def forward(self, g, h, e, eval_random=True):
        
        h = self.embedding_h(h)
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