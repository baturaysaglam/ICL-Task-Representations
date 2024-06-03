import sys

import torch

sys.path.append('..')

from src.FV_utils.eval_utils import *
from src.FV_utils.extract_utils import *


def get_attn_out(mean_activations, model, model_config):
    """
    Computes the output of multi-head attention using the mean activations of the individual attention heads.

    Parameters:
    mean_activations: tensor of size (Layers, Heads, Tokens, head_dim) containing the average activation of each head for a particular task
    model: huggingface model being used
    model_config: contains model config information (n layers, n heads, etc.)
    n_top_heads: The number of heads to use when computing the function vector

    Returns:
    attn_out: output of the MHA
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
