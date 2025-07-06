import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_add

class GCNConv(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.w = nn.Linear(node_dim, hidden_dim, bias=False)

    def forward(self, x, edge_index):
        num_nodes = x.shape[0]
        A = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        source = edge_index[0]
        target = edge_index[1]
        A[source, target] = 1
        A_tilde = A + torch.eye(num_nodes)

        D = A_tilde.sum(dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_tilde = torch.diag(D_inv_sqrt)

        out = D_tilde @ A_tilde @ D_tilde @ x
        out = self.w(out)
        return out


class GATConv(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(node_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(node_dim, hidden_dim, bias=False)
        self.a_s = nn.Parameter(torch.Tensor(hidden_dim)).unsqueeze(0)
        self.a_t = nn.Parameter(torch.Tensor(hidden_dim)).unsqueeze(0)
        nn.init.xavier_uniform_(self.a_s)
        nn.init.xavier_uniform_(self.a_t)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        self_loop_edges = torch.arange(x.shape[0])
        self_loop = torch.stack([self_loop_edges, self_loop_edges], dim=0)
        edge_index_with_loop = torch.cat([edge_index, self_loop], dim=1)

        source = edge_index_with_loop[0]
        target = edge_index_with_loop[1]

        x_j = x[source]
        x_i = x[target]
        wx_j = self.w1(x_j)
        wx_i = self.w2(x_i)

        e = self.leakyrelu(
        (self.a_s * wx_i).sum(dim=-1) + (self.a_t * wx_j).sum(dim=-1)
        )
        alpha = scatter_softmax(e, target)
        out = wx_j * alpha.unsqueeze(-1)
        out = scatter_add(out, target, dim=0, dim_size=x.shape[0])
        return out


class GCN_GAT(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.gcn = GCNConv(node_dim, hidden_dim)
        self.gat = GATConv(node_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        out1 = self.gcn(x, edge_index)
        out1 = torch.relu(out1)
        out2 = self.gat(x, edge_index)
        out2 = torch.relu(out2)

        out = out1 + out2
        out = self.linear(out)
        out = out.sum(dim=0)
        return out
