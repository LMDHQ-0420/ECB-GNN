import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
from torch_geometric.utils import softmax


class PreNormException(Exception):
    pass


class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super().__init__()
        assert shift or scale
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale

        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):

        assert self.n_units == 1 or input_.shape[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."

        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size())/self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg

        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False
        


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    def __init__(self, emb_size=64, heads=8, edge_in_dim=1, concat=True, negative_slope=0.2, attn_dropout=0.1):
        super().__init__(aggr='add')
        assert emb_size % heads == 0, "emb_size must be divisible by heads"
        self.emb_size = emb_size
        self.heads = heads
        self.head_dim = emb_size // heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.attn_dropout = attn_dropout

        self.feature_module_left = torch.nn.Linear(emb_size, emb_size)
        self.feature_module_edge = torch.nn.Linear(edge_in_dim, emb_size, bias=False)
        self.feature_module_right = torch.nn.Linear(emb_size, emb_size, bias=False)

        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(emb_size, shift=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )

        self.attn_proj = torch.nn.Linear(self.head_dim * 3 * self.heads, self.heads, bias=False)

        self.post_conv_module = torch.nn.Sequential(
            PreNormLayer(emb_size, shift=False)
        )

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features, index):
        E = node_features_i.size(0)

        left_proj = self.feature_module_left(node_features_i)
        edge_proj = self.feature_module_edge(edge_features)
        right_proj = self.feature_module_right(node_features_j)

        h_i = left_proj.view(E, self.heads, self.head_dim)
        h_j = right_proj.view(E, self.heads, self.head_dim)
        h_e = edge_proj.view(E, self.heads, self.head_dim)

        cat = torch.cat([h_i, h_j, h_e], dim=-1)
        cat_flat = cat.view(E, -1)
        scores = self.attn_proj(cat_flat)
        scores = scores.view(E, self.heads)

        scores = F.leaky_relu(scores, negative_slope=self.negative_slope)
        alphas = []
        for h in range(self.heads):
            alpha_h = softmax(scores[:, h], index)
            alpha_h = F.dropout(alpha_h, p=self.attn_dropout, training=self.training)
            alphas.append(alpha_h.unsqueeze(-1))
        alpha = torch.cat(alphas, dim=1)

        msg = (h_i + h_j + h_e)

        if self.concat:
            msg = msg.view(E, -1)
        else:
            msg = msg.mean(dim=1)

        msg = self.feature_module_final(msg)

        if self.concat:
            msg_heads = msg.view(E, self.heads, self.head_dim)
            msg_heads = msg_heads * alpha.unsqueeze(-1)
            msg = msg_heads.view(E, -1)
        else:
            msg = msg * alpha.mean(dim=1, keepdim=True)

        return msg

    def update(self, aggr_out):
        return aggr_out


class BaseModel(torch.nn.Module):

    def pre_train_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_train_next(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def pre_train(self, *args, **kwargs):
        try:
            with torch.no_grad():
                self.forward(*args, **kwargs)
            return False
        except PreNormException:
            return True


class GNNPolicy(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19

        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),
        )

        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        output = self.output_module(variable_features).squeeze(-1)
        return output