import numpy as np
import torch


def corrupt_prompt(xs, ys):
    shuffled_ys = ys[:, torch.randperm(ys.shape[1])]
    return xs, shuffled_ys


def compute_mean_act(model, xs, ys, L, j):
    model._backbone.h[L].track_attn_out = True
    _ = model(xs, ys)
    attn_out = model._backbone.h[L].attn_out
    model._backbone.h[L].track_attn_out = False
    mean_activation = attn_out[:, j, -2, :].mean(axis=0)
    mean_activation = mean_activation.unsqueeze(0)
    return mean_activation


def get_top_heads(model, xs, ys, num_layers, num_attn_heads, verbose=True):
    indirect_effect_mat = torch.zeros(num_attn_heads, num_layers).float().to(xs.device) * float('-inf')

    for L in range(num_layers):
        for j in range(num_attn_heads):
            # Compute the mean activation w.r.t. the true prompt sequence
            with torch.no_grad():
                mean_act = compute_mean_act(model, xs, ys, L, j)

                # Obtain the modified and original out given the corrupted data
                xs_corrupt, ys_corrupt = corrupt_prompt(xs, ys)

                pred_corrupt = model(xs_corrupt, ys_corrupt)[:, -1]
                modified_pred_corrupt = intervene_attn_act(model, xs_corrupt, ys_corrupt, mean_act, L, j)[:, -1]

            indirect_effect = indirect_metric(ys[:, -1], pred_corrupt, modified_pred_corrupt)
            indirect_effect_mat[j, L] = indirect_effect

            if verbose:
                print(f"For layer {L} and attn. head {j} - indirect effect: {indirect_effect:.3f}")

    return indirect_effect_mat


def get_universal_mean_act(model, xs, ys, n_layers, n_heads, head_dim):
    seq_len = int(xs.shape[1] * 2)
    mean_activations = torch.zeros((n_layers, n_heads, seq_len, head_dim)).to(xs.device)

    for L in range(n_layers):
        with torch.no_grad():
            model._backbone.h[L].track_attn_out = True
            _ = model(xs, ys)
            attn_out = model._backbone.h[L].attn_out
            model._backbone.h[L].track_attn_out = False
            mean_activations[L] = torch.mean(attn_out, axis=0)

    return mean_activations


def turn_on_causal_intervene(model, layer_idx, head_idx, mean_activation):
    model._backbone.h[layer_idx].intervene_layer = True
    model._backbone.h[layer_idx].intervene_head_idx = head_idx
    model._backbone.h[layer_idx].mean_attn_act = mean_activation

    return model


def turn_off_causal_intervene(model, layer_idx):
    model._backbone.h[layer_idx].intervene_layer = False
    model._backbone.h[layer_idx].intervene_head_idx = None
    model._backbone.h[layer_idx].mean_attn_act = None

    return model


def intervene_attn_act(model, xs, ys, mean_activation, L, j):
    # Turn on intervention
    turn_on_causal_intervene(model, L, j, mean_activation)

    pred = model(xs, ys)

    # Turn off intervention
    turn_off_causal_intervene(model, L)

    return pred


def distance(y, y_hat):
    return ((y - y_hat) ** 2).mean(axis=0)


def indirect_metric(ys, ys_hat, ys_hat_tilde):
    error = distance(ys, ys_hat)
    error_tilde = distance(ys, ys_hat_tilde)

    return error - error_tilde


def top_heads_locations(matrix, num_top_heads):
    # Flatten the matrix and get the indices of the top values
    flat_indices = np.argpartition(-matrix.flatten(), range(num_top_heads))[:num_top_heads]

    # Convert flat indices to two-dimensional indices
    top_indices = np.array(np.unravel_index(flat_indices, matrix.shape)).T

    # Sort the indices by the value in descending order
    top_values_and_indices = [(matrix[tuple(idx)], tuple(idx)) for idx in top_indices]
    top_values_and_indices.sort(reverse=True, key=lambda x: x[0])

    # Extract the sorted indices
    sorted_top_indices = [idx for _, idx in top_values_and_indices]

    return sorted_top_indices


def find_indices_larger_than(matrix, threshold):
    mask = matrix > threshold
    indices = np.where(mask)

    result = list(zip(indices[0], indices[1]))

    return result


def translate_transformer_output(true_model, model_out, out_dim):
    pred = true_model._read_out(model_out.last_hidden_state)
    pred = pred[:, ::2, 0][:, torch.arange(out_dim)]
    return pred


def compute_function_vector(model, mean_activations, top_heads, resid_dim, head_dim):
    device = mean_activations.device
    function_vector = torch.zeros((1, 1, resid_dim)).to(device)
    T = -2  # Intervention & values taken from the last token

    for H, L in top_heads:
        out_proj = model._backbone.h[L].attn.c_proj

        x = torch.zeros(resid_dim)
        x[H * head_dim:(H + 1) * head_dim] = mean_activations[L, H, T]
        d_out = out_proj(x.reshape(1, 1, resid_dim).to(device).to(model._backbone.dtype))

        function_vector += d_out
        function_vector = function_vector.to(model._backbone.dtype)

    function_vector = function_vector.reshape(1, resid_dim)

    return function_vector


def add_function_vector(model, xs, ys, L, FV, dummy_indices=None, scale=1, output_hidden_states=False):
    """
    Modify the output of a specified layer in a transformer model by adding a vector.

    :param model: The transformer model.
    :param input_data: The input data to the model.
    :param L: The layer number whose output will be modified.
    :param vector_to_add: The vector to be added to the output of the specified layer.
    :return: The modified output of the transformer.
    """
    if isinstance(L, (int, np.integer)):
        L = [L]

    if len(L) > 1:
            scale = scale * (1 / len(L))

    # A hook function that adds the vector to the output of the specified layers.
    def modify_layer_output(layer_index):
        def hook(module, input, output):
            # Check if the current layer is in the specified layer list
            if layer_index in L:
                # Add the modifier vector to the output of this layer
                output[0][:, -2] += scale * FV

                # if dummy_indices is not None:
                if dummy_indices is not None:
                    for dummy_idx in dummy_indices:
                        dummy_idx_seq = max(int((dummy_idx - 1) * 2), 0)
                        output[0][:, dummy_idx_seq] += scale * FV
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
