import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter
import torch.nn as nn
from greatx.nn.layers import GCNConv, Sequential, activations
from greatx.utils import wrapper


EPS = 1e-10


class MyLayer(torch.nn.Module):
    def __init__(self, mask, add_self_loops=True):
        super().__init__()
        self.add_self_loops = add_self_loops
        self.mask = mask

    def forward(self, x, edge_index):
        row, col = edge_index
        A, B = x[row], x[col]
        att_score = F.cosine_similarity(A, B)

        edge_index = edge_index[:, self.mask]
        att_score = att_score[self.mask]

        row, col = edge_index
        row_sum = scatter(att_score, col, dim_size=x.size(0))
        att_score_norm = att_score / (row_sum[row] + EPS)

        if self.add_self_loops:
            degree = scatter(torch.ones_like(att_score_norm), col, dim_size=x.size(0))
            self_weight = 1.0 / (degree + 1)
            att_score_norm = torch.cat([att_score_norm, self_weight])
            loop_index = torch.arange(
                0, x.size(0), dtype=torch.long, device=edge_index.device
            )
            loop_index = loop_index.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, loop_index], dim=1)

        att_score_norm = att_score_norm.exp()
        return edge_index, att_score_norm

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"


class MyModel(nn.Module):
    @wrapper
    def __init__(self, in_channels, out_channels, normalize, bias, mask):
        super().__init__()

        conv = []
        conv.append(MyLayer(mask=mask, add_self_loops=True))
        conv.append(
            GCNConv(
                in_channels,
                out_channels,
                add_self_loops=False,
                bias=bias,
                normalize=normalize,
            )
        )
        self.conv = Sequential(*conv)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        for layer in self.conv:
            if isinstance(layer, MyLayer):
                edge_index, edge_weight = layer(x, edge_index)
            elif isinstance(layer, GCNConv):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x
