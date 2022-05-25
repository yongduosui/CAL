import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from layers.gin_layer import GINLayer, ApplyNodeFunc, MLP
from layers.causal_readout_layer import CausalReadout_complex
import pdb

class CausalGIN_complex(nn.Module):
    
    def __init__(self, net_params, args):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        self.n_classes = n_classes
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        n_mlp_layers = net_params['n_mlp_GIN']               # GIN
        learn_eps = net_params['learn_eps_GIN']              # GIN
        neighbor_aggr_type = net_params['neighbor_aggr_GIN'] # GIN
        readout = net_params['readout']                      # this is graph_pooling_type     
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']     
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        for layer in range(self.n_layers):
            mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, batch_norm, residual, 0, learn_eps))
  
        self.causal_layer = torch.nn.ModuleList()
        self.pool = SumPooling()
        for layer in range(self.n_layers+1):
            self.causal_layer.append(CausalReadout_complex(hidden_dim, n_classes, args, self.pool))
        
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(self.n_layers+1):
            self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))

    def forward(self, g, h, e, eval_random=True):
        
        h = self.embedding_h(h)
        hidden_rep = [h]

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h)
            hidden_rep.append(h)
        score_over_layer = 0
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.linears_prediction[i](pooled_h)
        
        num_rep = len(hidden_rep)
        xc, xo, xco = self.causal_layer[i](g, hidden_rep[-1], eval_random)
        xo += score_over_layer
        xco += score_over_layer
        return xc, xo, xco
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss

    def kl_loss(self, pred, label):
        xco_logis = F.log_softmax(pred, dim=-1)
        loss = F.kl_div(xco_logis, label, reduction='batchmean')
        return loss