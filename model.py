from functools import partial
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GINConv, GATConv
from gcn_conv import GCNConv
import random
import pdb
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import os

class CausalGCN(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                gfn=False, collapse=False, residual=False,
                res_branch="BNConvReLU", global_pool="sum", dropout=0, 
                edge_norm=True):
        super(CausalGCN, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, vis=False):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))
        
        edge_rep = torch.cat([x[row], x[col]], dim=-1)

        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2).cuda()
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        if self.without_node_attention:
            node_att = 0.5 * torch.ones(x.shape[0], 2).cuda()
        else:
            node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        xc = node_att[:, 0].view(-1, 1) * x
        xo = node_att[:, 1].view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        xc = self.global_pool(xc, batch)
        xo = self.global_pool(xo, batch)
        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)

        return xc_logis, xo_logis, xco_logis


    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if self.with_random:
            if eval_random:
                random.shuffle(l)
        random_idx = torch.tensor(l)
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

class CausalGIN(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                gfn=False,
                edge_norm=True):
        super(CausalGIN, self).__init__()

        hidden = args.hidden
        num_conv_layers = args.layers
        self.args = args
        self.global_pool = global_add_pool
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.convs.append(GINConv(
            Sequential(
                       Linear(hidden, hidden), 
                       BatchNorm1d(hidden), 
                       ReLU(),
                       Linear(hidden, hidden), 
                       ReLU())))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, vis=False, train_type="base"):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
        
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        node_weight_c = node_att[:, 0]
        node_weight_o = node_att[:, 1]
        
        if vis:
            '''
            BA + Cycle   48
            BA + Grid    50
            BA + Diamond 82
            BA + House   60
            
            Tr + Cycle   78
            Tr + Grid    87
            Tr + Diamond 98
            Tr + House   89
            
            '''
            # index = [48, 50, 98, 89]
            # index = [82, 60, 78, 87]
            # node_attention = [node_weight_c, node_weight_o]
            # edge_attention = [edge_weight_c, edge_weight_o]
            # plot_attention_graph(data, [node_weight_c, node_weight_o], [edge_weight_c, edge_weight_o], index=87)
            path = "visual-{}-bias{}".format(self.args.model, self.args.bias[0])
            folder = os.path.exists(path)
            if not folder:
                os.makedirs(path)
            for i in range(80):
                print("plot num:{}".format(i))
                plot_attention_graph(data, node_weight_o, edge_weight_o, path, index=i)

            return None, None, None

        xc = node_weight_c.view(-1, 1) * x
        xo = node_weight_o.view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        xc = self.global_pool(xc, batch)
        xo = self.global_pool(xo, batch)
        
        xc_logis = self.context_readout_layer(xc)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)
        # return xc_logis, xo_logis, xco_logis
        if train_type == "base":
            xo_logis = self.objects_readout_layer(xo, train_type)
            return xc_logis, xo_logis, xco_logis
        elif train_type == "irm":
            xo_before, xo_logis = self.objects_readout_layer(xo, train_type)
            return xc_logis, xo_before, xo_logis, xco_logis
        else:
            assert False


    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x, train_type):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        if train_type == "irm":
            return x, x_logis
        else:
            return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if eval_random:
            random.shuffle(l)
        random_idx = torch.tensor(l)
        
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis



def plot_four_attention_graph(data, node_attention, edge_attention, index_list):

    plt.figure(figsize=(8, 10))
    plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.05)
    plt.subplot(221)
    plot_one_graph(data, node_attention, edge_attention, index_list[0])
    plt.subplot(222)
    plot_one_graph(data, node_attention, edge_attention, index_list[1])
    plt.subplot(223)
    plot_one_graph(data, node_attention, edge_attention, index_list[2])
    plt.subplot(224)
    plot_one_graph(data, node_attention, edge_attention, index_list[3])
    plt.tight_layout()
    plt.savefig("vis-paper{}.png".format(0))
    plt.close()

def plot_one_graph(data, node_attention, edge_attention, index):

    data_list = data.to_data_list()
    num_graphs = len(data_list)
    total_nodes = 0
    total_edges = 0
    for g in data_list[:index]:
        total_nodes += g.num_nodes
        total_edges += g.num_edges
    
    graph = data_list[index]
    node_num = graph.num_nodes
    edge_num = graph.num_edges
    node_att_o = node_attention[1][total_nodes:total_nodes+node_num]
    edge_att_o = edge_attention[1][total_edges:total_edges+edge_num]
    edge_color = edge_att_o.tolist()
    node_color = node_att_o.tolist()
    G = to_networkx(graph)
    options = {"edgecolors": "tab:grey", "alpha": 1.0}
    nx.draw(G, 
            pos=nx.kamada_kawai_layout(G),
            cmap="Blues",
            node_color=node_color,
            edge_color="grey",
            width=edge_color,
            node_size=200, 
            arrows=False,
            **options)

def plot_attention_graph(data, node_attention, edge_attention, path, index):

    data_list = data.to_data_list()
    num_graphs = len(data_list)
    total_nodes = 0
    total_edges = 0
    for g in data_list[:index]:
        total_nodes += g.num_nodes
        total_edges += g.num_edges
    
    graph = data_list[index]
    node_num = graph.num_nodes
    edge_num = graph.num_edges
    node_att_o = node_attention[total_nodes:total_nodes+node_num]
    edge_att_o = edge_attention[total_edges:total_edges+edge_num]
    
    node_att_o = F.softmax(node_att_o)
    edge_color = edge_att_o.tolist()
    # pdb.set_trace()
    node_color = node_att_o.tolist()
    G = to_networkx(graph)
    plt.figure(figsize=(7, 7))
    plt.axis('off')
    options = {"edgecolors": "tab:grey", "alpha": 1.0}
    # pdb.set_trace()
    nx.draw(G, pos=nx.kamada_kawai_layout(G),
               cmap="Blues",
               node_color=node_color,
               edge_color="grey",
               vmin = node_att_o.max() * 0.5,
               vmax = node_att_o.max(),
               width=edge_color,
               node_size=200, 
               arrows=False,
               **options)
    
    plt.tight_layout()
    plt.savefig(path + "/causal-vis-{}.png".format(index))
    plt.close()

# def plot_attention_graph(data, node_attention, edge_attention, index):

#     data_list = data.to_data_list()
#     num_graphs = len(data_list)
#     total_nodes = 0
#     total_edges = 0
#     for g in data_list[:index]:
#         total_nodes += g.num_nodes
#         total_edges += g.num_edges
    
#     graph = data_list[index]
#     node_num = graph.num_nodes
#     edge_num = graph.num_edges
#     node_att_o = node_attention[1][total_nodes:total_nodes+node_num]
#     edge_att_o = edge_attention[1][total_edges:total_edges+edge_num]
#     edge_color = edge_att_o.tolist()
#     node_color = node_att_o.tolist()
#     G = to_networkx(graph)
#     plt.figure(figsize=(7, 7))
#     plt.axis('off')
#     options = {"edgecolors": "tab:grey", "alpha": 1.0}
#     # pdb.set_trace()
#     nx.draw(G, pos=nx.kamada_kawai_layout(G),
#                cmap="Blues",
#                node_color=node_color,
#                edge_color="grey",
#                width=edge_color,
#                node_size=200, 
#                arrows=False,
#                **options)
    
#     plt.tight_layout()
#     plt.savefig("causal-vis-{}.png".format(index))
#     plt.close()


class CausalGAT(torch.nn.Module):
    def __init__(self, num_features,
                       num_classes, 
                       args, 
                       head=4, 
                       dropout=0.2):
        super(CausalGAT, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=True, gfn=False)

        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))
        
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        xc = node_att[:, 0].view(-1, 1) * x
        xo = node_att[:, 1].view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        xc = self.global_pool(xc, batch)
        xo = self.global_pool(xo, batch)
        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)

        return xc_logis, xo_logis, xco_logis


    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if eval_random:
            random.shuffle(l)
        random_idx = torch.tensor(l)
        
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis