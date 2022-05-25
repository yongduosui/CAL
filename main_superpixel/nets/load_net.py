from nets.gcn_net import GCNNet
from nets.gat_net import GATNet
from nets.gin_net import GINNet
from nets.causal_gcn_net_complex import CausalGCN_complex
from nets.causal_gin_net_complex import CausalGIN_complex
from nets.causal_gat_net_complex import CausalGAT_complex
import pdb
def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GIN(net_params):
    return GINNet(net_params)


def gnn_model(MODEL_NAME, net_params, args=None):
    models = {
        'GCN': GCN,
        'GAT': GAT,
        'GIN': GIN,
        'CausalGCN': CausalGCN_complex,
        'CausalGIN': CausalGIN_complex,
        'CausalGAT': CausalGAT_complex
    }
    if 'Causal' in MODEL_NAME:
        return models[MODEL_NAME](net_params, args)
    else:
        return models[MODEL_NAME](net_params)