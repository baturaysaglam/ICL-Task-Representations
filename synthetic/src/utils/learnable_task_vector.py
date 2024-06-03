import torch
import torch.nn as nn

from .function_vector import add_function_vector


def swish(x, beta=1):
    return x * torch.sigmoid(beta * x)


class LearnableTaskVector(nn.Module):
    """
    Class for Learnable Task vectors.
    No activation functions or further optimization technique is used.
    The weights are initialized to be between -1 and 1 according to the standard Gaussian distribution.
    """
    def __init__(self, n_layers, n_heads, act_fn=None):
        super(LearnableTaskVector, self).__init__()
        self.weights = nn.Parameter(torch.randn(n_layers, n_heads))
        self.act_fn = act_fn

    def forward(self, x):
        # Apply sigmoid to constrain weights between 0 and 1
        # normalized_weights = torch.sigmoid(self.weights).unsqueeze(-1)
        normalized_weights = self.weights.unsqueeze(-1)

        # Weighted sum across n_heads
        weighted_sum = torch.sum(normalized_weights * x, dim=2)

        # Apply activation function if not None
        if self.act_fn == 'leaky_relu':
            weighted_sum = nn.LeakyReLU()(weighted_sum)
        elif self.act_fn == 'swish':
            weighted_sum = swish(weighted_sum)

        return weighted_sum


def get_attn_outs(model, xs, ys, n_layers, n_heads, head_dim, resid_dim, type=None, LTV=None, FV=None, FV_layers=None, dummy_indices=None, scale=1.0):
    batch_size = xs.shape[0]
    mean_activations = torch.zeros((batch_size, n_layers, n_heads, head_dim)).to(xs.device)

    T = -2  # Intervention & values taken from the last token

    for L in range(n_layers):
        with torch.no_grad():
            model._backbone.h[L].track_attn_out = True

            if type == 'LTV':
                assert LTV is not None
                _ = add_learnable_task_vector(model, xs, ys, LTV=LTV, dummy_indices=dummy_indices, scale=1.0)
            elif type == 'FV':
                assert FV is not None
                _ = add_function_vector(model, xs, ys, L=FV_layers, FV=FV, dummy_indices=dummy_indices, scale=scale)
            else:
                _ = model(xs, ys)

            attn_out = model._backbone.h[L].attn_out
            model._backbone.h[L].track_attn_out = False
            mean_activations[:, L, :, :] = attn_out[:, :, T, :]

    attn_outs = torch.zeros(batch_size, n_layers, n_heads, resid_dim).to(model._backbone.device)

    for layer_idx, L in enumerate(range(n_layers)):
        for head_idx, H in enumerate(range(n_heads)):
            out_proj = model._backbone.h[L].attn.c_proj

            x = torch.zeros(batch_size, resid_dim)
            x[:, H * head_dim:(H + 1) * head_dim] = mean_activations[:, L, H]
            d_out = out_proj(x.reshape(batch_size, 1, resid_dim).to(model._backbone.device).to(model._backbone.dtype))
            attn_outs[:, layer_idx, head_idx, :] = d_out.squeeze()

    return attn_outs


def get_activations(model, xs, ys, n_layers, n_heads, head_dim, resid_dim, type=None, LTV=None, FV=None, FV_layers=None, dummy_indices=None, scale=1.0):
    batch_size = xs.shape[0]
    mean_activations = torch.zeros((batch_size, n_layers, n_heads, head_dim)).to(xs.device)

    T = -2  # Intervention & values taken from the last token

    for L in range(n_layers):
        with torch.no_grad():
            model._backbone.h[L].track_attn_out = True

            if type == 'LTV':
                assert LTV is not None
                _, output = add_learnable_task_vector(model, xs, ys, LTV=LTV, dummy_indices=dummy_indices, scale=1.0, output_hidden_states=True)
            elif type == 'FV':
                assert FV is not None
                _, output = add_function_vector(model, xs, ys, L=FV_layers, FV=FV, dummy_indices=dummy_indices, scale=scale, output_hidden_states=True)
            else:
                _, output = model.get_transformer_out(xs, ys)

            attn_out = model._backbone.h[L].attn_out
            model._backbone.h[L].track_attn_out = False
            mean_activations[:, L, :, :] = attn_out[:, :, T, :]

    attn_outs = torch.zeros(batch_size, n_layers, n_heads, resid_dim).to(model._backbone.device)

    for layer_idx, L in enumerate(range(n_layers)):
        for head_idx, H in enumerate(range(n_heads)):
            out_proj = model._backbone.h[L].attn.c_proj

            x = torch.zeros(batch_size, resid_dim)
            x[:, H * head_dim:(H + 1) * head_dim] = mean_activations[:, L, H]
            d_out = out_proj(x.reshape(batch_size, 1, resid_dim).to(model._backbone.device).to(model._backbone.dtype))
            attn_outs[:, layer_idx, head_idx, :] = d_out.squeeze()

    hidden_states = output.hidden_states
    hidden_states = [state[:, T, :] for state in hidden_states]
    hidden_states = torch.stack(hidden_states).to(hidden_states[0].device)
    hidden_states = torch.transpose(hidden_states, 0, 1)

    return attn_outs, hidden_states


def add_learnable_task_vector(model, xs, ys, LTV, dummy_indices=None, scale=1.0, output_hidden_states=False):
    """
    Runs the model on the sentence and adds the lt_vector to the output of layer activations as a model intervention, predicting a single token.
    Returns the output of the model with intervention.

    Parameters:
    model: huggingface model
    xs: covariates
    ys: labels --> ys = f(xs)
    LTV: torch vector that triggers execution of a task
    dummy_indices: position of dummy variables to "remind" the model of the task
    scale: scale of the LTV
    output_hidden_states: to output the hidden states or just the modified outout

    Returns:
    modified_output: a tuple containing output results of a clean run and intervened run of the model
    """
    # A hook function that adds the vector to the output of the specified layers.
    def modify_layer_output(layer_idx):
        def hook(module, input, output):
            # Add the modifier vector to the output of this layer
            output[0][:, -2] += scale * LTV[:, layer_idx, :]

            # # if dummy_indices is not None:
            if dummy_indices is not None:
                for dummy_idx in dummy_indices:
                    dummy_idx_seq = max(int((dummy_idx - 1) * 2), 0)
                    output[0][:, dummy_idx_seq] += scale * LTV[:, layer_idx, :]
            return output

        return hook

    # Register a forward hook for the specified layers
    handles = []
    for i, layer in enumerate(model._backbone.h):
        handle = layer.register_forward_hook(modify_layer_output(i))
        handles.append(handle)

    # Pass the input through the model
    if output_hidden_states:
        modified_output = model.get_transformer_out(xs, ys)
    else:
        modified_output = model(xs, ys)

    # Remove all forward hooks
    for handle in handles:
        handle.remove()

    return modified_output


def data_sampler(model, conf, task_sampler, covariate_sampler, seq_len, batch_size, n_layers, n_heads, head_dim,
                 resid_dim, device):
    pretrain_seq_len = conf.training.curriculum.points.end
    task = task_sampler()
    xs_long = covariate_sampler.sample_xs(b_size=batch_size, n_points=seq_len).to(device)
    ys_long = task.evaluate(xs_long).to(device)

    xs, ys = xs_long[:, :pretrain_seq_len, :], ys_long[:, :pretrain_seq_len]
    label = ys_long

    attn_out = get_attn_outs(model, xs, ys, n_layers, n_heads, head_dim, resid_dim)

    return attn_out, label, xs_long, ys_long
