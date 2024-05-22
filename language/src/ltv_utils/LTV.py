import torch
import torch.nn as nn


class LearnableTaskVector(nn.Module):
    def __init__(self, n_layers, n_heads, n_head_dim, act_fn=None):
        super(LearnableTaskVector, self).__init__()
        # Initialize the weights using a uniform distribution between 0 and 1
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_head_dim = n_head_dim
        self.weights = nn.Parameter(torch.randn(n_layers, n_heads))
        self.act_fn = act_fn

    def forward(self, x):
        # Reshape weight to shape [1, n_layers, n_heads, 1]
        batch_size = x.shape[0]
        normalized_weights = self.weights.unsqueeze(0).unsqueeze(-1)

        weighted_sum = x * normalized_weights

        # The result will have shape [batch_size, n_layers, n_heads * n_head_dim]
        out = weighted_sum.view(batch_size, self.n_layers, self.n_heads * self.n_head_dim)

        return out