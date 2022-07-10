from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GINConv, GATConv
from gcn_conv import GCNConv
import random
import pdb

class CausalGCN(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                       gfn=False, 
                       collapse=False, 
                       residual=False,
                       res_branch="BNConvReLU", 
                       global_pool="sum", 
                       dropout=0, 
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

    def forward(self, data, eval_random=True, train_type="base"):

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
        
        
        xc = node_weight_c.view(-1, 1) * x
        xo = node_weight_o.view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        xc = self.global_pool(xc, batch)
        xo = self.global_pool(xo, batch)
        
        xc_logis = self.context_readout_layer(xc)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)
        # return xc_logis, xo_logis, xco_logis
        xo_logis = self.objects_readout_layer(xo, train_type)
        return xc_logis, xo_logis, xco_logis
        


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

class GCNNet(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, hidden, 
                       num_feat_layers=1, 
                       num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0, 
                 edge_norm=True):
        super(GCNNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))
            
        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

class GINNet(torch.nn.Module):
    def __init__(self, num_features,
                       num_classes,
                       hidden, 
                       num_fc_layers=2, 
                       num_conv_layers=3, 
                       dropout=0):

        super(GINNet, self).__init__()
        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden_in = num_features
        hidden_out = num_classes
        
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.convs.append(GINConv(
            Sequential(Linear(hidden, hidden), 
                       BatchNorm1d(hidden), 
                       ReLU(),
                       Linear(hidden, hidden), 
                       ReLU())))

        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        # x, edge_index, batch = data.feat, data.edge_index, data.batch
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))    
        x = self.bn_hidden(x)
        x = self.lin_class(x)

        prediction = F.log_softmax(x, dim=-1)
        return prediction

class GATNet(torch.nn.Module):
    def __init__(self, num_features, 
                       num_classes,
                       hidden,
                       head=4,
                       num_fc_layers=2, 
                       num_conv_layers=3, 
                       dropout=0.2):

        super(GATNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden_in = num_features
        hidden_out = num_classes
   
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))

        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)
