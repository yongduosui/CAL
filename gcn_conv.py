
import torch
# from typing import Union, Tuple, Optional
# from torch_geometric.typing import (Adj, OptTensor, PairTensor)
# from torch import Tensor
# import torch.nn.functional as F
# from torch_geometric.nn.dense.linear import Linear 
# from torch_sparse import SparseTensor, set_diag
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros, reset
import pdb



class GCNConv(MessagePassing):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 edge_norm=True,
                 gfn=False):
        super(GCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.edge_norm = edge_norm
        self.gfn = gfn
        self.message_mask = None
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        
        edge_weight = edge_weight.view(-1)
        
        
        assert edge_weight.size(0) == edge_index.size(1)
        
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes, ),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        
        x = torch.matmul(x, self.weight)
        if self.gfn:
            return x
    
        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = GCNConv.norm(
                    edge_index, 
                    x.size(0), 
                    edge_weight, 
                    self.improved, 
                    x.dtype)
            else:
                norm = None
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):

        if self.edge_norm:
            return norm.view(-1, 1) * x_j
        else:
            return x_j
        
    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
    # def message(self, x_j, norm):
    #     if self.message_mask is None:
    #         if self.edge_norm:
    #             return norm.view(-1, 1) * x_j
    #         else:
    #             return x_j
    #     else:
    #         if self.edge_norm:
                
    #             link_num = self.message_mask.shape[0]
    #             out_message = norm.view(-1, 1) * x_j
    #             out_message[:link_num] = out_message[:link_num] * self.message_mask.view(-1, 1)
    #             return out_message
    #         else:
    #             link_num = self.message_mask.shape[0]
    #             x_j[:link_num] = x_j[:link_num] * self.message_mask
    #             return x_j


class GlobalAttention(torch.nn.Module):
    
    def __init__(self, gate_nn, nn=None):
        super(GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None, vis=False):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)
        if vis:
            return out, gate
        else:
            return out

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)


class AGNNConv(MessagePassing):
    
    def __init__(self, requires_grad=True, **kwargs):
        super(AGNNConv, self).__init__(aggr='add', **kwargs)

        self.requires_grad = requires_grad

        if requires_grad:
            self.beta = Parameter(torch.Tensor(1))
        else:
            self.register_buffer('beta', torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.requires_grad:
            self.beta.data.fill_(1)

    def forward(self, x, edge_index):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x.size(self.node_dim))

        x_norm = F.normalize(x, p=2, dim=-1)

        return self.propagate(edge_index, 
                              x=x, 
                              x_norm=x_norm,
                              num_nodes=x.size(self.node_dim))

    def message(self, edge_index_i, x_j, x_norm_i, x_norm_j, num_nodes):
        # Compute attention coefficients.
        beta = self.beta if self.requires_grad else self._buffers['beta']
        alpha = beta * (x_norm_i * x_norm_j).sum(dim=-1)
        alpha = softmax(alpha, edge_index_i, num_nodes)

        return x_j * alpha.view(-1, 1)

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


# class GATv2Conv(MessagePassing):
    
#     _alpha: OptTensor

#     def __init__(
#         self,
#         in_channels: Union[int, Tuple[int, int]],
#         out_channels: int,
#         heads: int = 1,
#         concat: bool = True,
#         negative_slope: float = 0.2,
#         dropout: float = 0.0,
#         add_self_loops: bool = True,
#         edge_dim: Optional[int] = None,
#         fill_value: Union[float, Tensor, str] = 'mean',
#         bias: bool = True,
#         share_weights: bool = False,
#         **kwargs,
#     ):
#         super().__init__(node_dim=0, **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.heads = heads
#         self.concat = concat
#         self.negative_slope = negative_slope
#         self.dropout = dropout
#         self.add_self_loops = add_self_loops
#         self.edge_dim = edge_dim
#         self.fill_value = fill_value
#         self.share_weights = share_weights

#         if isinstance(in_channels, int):
#             self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
#                                 weight_initializer='glorot')
#             if share_weights:
#                 self.lin_r = self.lin_l
#             else:
#                 self.lin_r = Linear(in_channels, heads * out_channels,
#                                     bias=bias, weight_initializer='glorot')
#         else:
#             self.lin_l = Linear(in_channels[0], heads * out_channels,
#                                 bias=bias, weight_initializer='glorot')
#             if share_weights:
#                 self.lin_r = self.lin_l
#             else:
#                 self.lin_r = Linear(in_channels[1], heads * out_channels,
#                                     bias=bias, weight_initializer='glorot')

#         self.att = Parameter(torch.Tensor(1, heads, out_channels))

#         if edge_dim is not None:
#             self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
#                                    weight_initializer='glorot')
#         else:
#             self.lin_edge = None

#         if bias and concat:
#             self.bias = Parameter(torch.Tensor(heads * out_channels))
#         elif bias and not concat:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self._alpha = None

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin_l.reset_parameters()
#         self.lin_r.reset_parameters()
#         if self.lin_edge is not None:
#             self.lin_edge.reset_parameters()
#         glorot(self.att)
#         zeros(self.bias)


#     def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
#                 edge_attr: OptTensor = None,
#                 return_attention_weights: bool = None):
        
#         H, C = self.heads, self.out_channels

#         x_l: OptTensor = None
#         x_r: OptTensor = None
#         if isinstance(x, Tensor):
#             assert x.dim() == 2
#             x_l = self.lin_l(x).view(-1, H, C)
#             if self.share_weights:
#                 x_r = x_l
#             else:
#                 x_r = self.lin_r(x).view(-1, H, C)
#         else:
#             x_l, x_r = x[0], x[1]
#             assert x[0].dim() == 2
#             x_l = self.lin_l(x_l).view(-1, H, C)
#             if x_r is not None:
#                 x_r = self.lin_r(x_r).view(-1, H, C)

#         assert x_l is not None
#         assert x_r is not None

#         if self.add_self_loops:
#             if isinstance(edge_index, Tensor):
#                 num_nodes = x_l.size(0)
#                 if x_r is not None:
#                     num_nodes = min(num_nodes, x_r.size(0))
#                 edge_index, edge_attr = remove_self_loops(
#                     edge_index, edge_attr)
#                 edge_index, edge_attr = add_self_loops(
#                     edge_index, edge_attr, fill_value=self.fill_value,
#                     num_nodes=num_nodes)
#             elif isinstance(edge_index, SparseTensor):
#                 if self.edge_dim is None:
#                     edge_index = set_diag(edge_index)
#                 else:
#                     raise NotImplementedError(
#                         "The usage of 'edge_attr' and 'add_self_loops' "
#                         "simultaneously is currently not yet supported for "
#                         "'edge_index' in a 'SparseTensor' form")

#         # propagate_type: (x: PairTensor, edge_attr: OptTensor)
#         out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
#                              size=None)

#         alpha = self._alpha
#         self._alpha = None

#         if self.concat:
#             out = out.view(-1, self.heads * self.out_channels)
#         else:
#             out = out.mean(dim=1)

#         if self.bias is not None:
#             out += self.bias

#         if isinstance(return_attention_weights, bool):
#             assert alpha is not None
#             if isinstance(edge_index, Tensor):
#                 return out, (edge_index, alpha)
#             elif isinstance(edge_index, SparseTensor):
#                 return out, edge_index.set_value(alpha, layout='coo')
#         else:
#             return out


#     def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
#                 index: Tensor, ptr: OptTensor,
#                 size_i: Optional[int]) -> Tensor:
#         x = x_i + x_j

#         if edge_attr is not None:
#             if edge_attr.dim() == 1:
#                 edge_attr = edge_attr.view(-1, 1)
#             assert self.lin_edge is not None
#             edge_attr = self.lin_edge(edge_attr)
#             edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
#             x += edge_attr

#         x = F.leaky_relu(x, self.negative_slope)
#         alpha = (x * self.att).sum(dim=-1)
#         alpha = softmax(alpha, index, ptr, size_i)
#         self._alpha = alpha
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#         return x_j * alpha.unsqueeze(-1)

#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.in_channels}, '
#                 f'{self.out_channels}, heads={self.heads})')