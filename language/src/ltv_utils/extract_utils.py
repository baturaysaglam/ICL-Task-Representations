import torch

from src.utils.eval_utils import *
from src.utils.extract_utils import *


def get_attn_out(mean_activations, model, model_config):
    """
        Computes a "function vector" vector that communicates the task observed in ICL examples used for downstream intervention
        using the set of heads with universally highest causal effect computed across a set of ICL tasks

        Parameters:
        mean_activations: tensor of size (Layers, Heads, Tokens, head_dim) containing the average activation of each head for a particular task
        model: huggingface model being used
        model_config: contains model config information (n layers, n heads, etc.)
        n_top_heads: The number of heads to use when computing the function vector

        Returns:
        function_vector: vector representing the communication of a particular task
        top_heads: list of the top influential heads represented as tuples [(L,H,S), ...], (L=Layer, H=Head, S=Avg. Indirect Effect Score)
    """
    model_n_layers = model_config['n_layers']
    model_resid_dim = model_config['resid_dim']
    model_n_heads = model_config['n_heads']
    model_head_dim = model_resid_dim // model_n_heads

    if len(mean_activations.shape) == 4:
        mean_activations = mean_activations.unsqueeze(0)

    batch_size = mean_activations.shape[0]
    device = model.device

    mean_activations = mean_activations[:, :, :, -1]  # Intervention & values taken from last token
    attn_out = torch.zeros((batch_size, model_n_layers, model_n_heads, model_head_dim)).to(device)

    for L in range(model_n_layers):
        if 'gpt2-xl' in model_config['name_or_path']:
            out_proj = model.transformer.h[L].attn.c_proj
        elif 'gpt-j' in model_config['name_or_path']:
            out_proj = model.transformer.h[L].attn.out_proj
        elif 'llama' in model_config['name_or_path']:
            out_proj = model.model.layers[L].self_attn.o_proj
        elif 'gpt-neox' in model_config['name_or_path']:
            out_proj = model.gpt_neox.layers[L].attention.dense

        x = mean_activations[:, L, :]
        x_reshaped = x.reshape(batch_size, 1, model_resid_dim).to(device)
        d_out = out_proj(x_reshaped)
        attn_out[:, L, :] = d_out.view(batch_size, model_n_heads, model_head_dim)

    return attn_out


def get_head_activations_on_prompt(prompt_data, model, model_config, tokenizer, n_icl_examples=10,
                                   shuffle_labels=False, prefixes=None, separators=None, filter_set=None):
    """
    Computes the average activations for each attention head in the model, where multi-token phrases are condensed into a single slot through averaging.

    Parameters:
    dataset: ICL dataset
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_icl_examples: Number of shots in each in-context prompt
    N_TRIALS: Number of in-context prompts to average over
    shuffle_labels: Whether to shuffle the ICL labels or not
    prefixes: ICL template prefixes
    separators: ICL template separators
    filter_set: whether to only include samples the model gets correct via ICL

    Returns:
    mean_activations: avg activation of each attention head in the model taken across n_trials ICL prompts
    """

    def split_activations_by_head(activations, model_config):
        new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim'] // model_config[
            'n_heads'])  # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
        activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
        return activations

    n_test_examples = 1
    if prefixes is not None and separators is not None:
        dummy_labels = get_dummy_token_labels(n_icl_examples, tokenizer=tokenizer, prefixes=prefixes,
                                              separators=separators)
    else:
        dummy_labels = get_dummy_token_labels(n_icl_examples, tokenizer=tokenizer)

    activations_td, idx_map, idx_avg = gather_attn_activations(prompt_data=prompt_data,
                                                               layers=model_config['attn_hook_names'],
                                                               dummy_labels=dummy_labels,
                                                               model=model,
                                                               tokenizer=tokenizer)

    stack_initial = torch.vstack([split_activations_by_head(activations_td[layer].input, model_config) for layer in
                                  model_config['attn_hook_names']]).permute(0, 2, 1, 3).to(model.device)
    stack_filtered = stack_initial[:, :, list(idx_map.keys())]
    for (i, j) in idx_avg.values():
        stack_filtered[:, :, idx_map[i]] = stack_initial[:, :, i:j + 1].mean(axis=2)  # Average activations of multi-token words across all its tokens

    mean_activations = stack_filtered.unsqueeze(0)

    return mean_activations