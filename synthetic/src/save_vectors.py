import argparse
import os

import torch.nn.functional as F
import torch.linalg as LA

from utils.experiment import *
from utils.learnable_task_vector import *
from utils.plot import *


n_FV_heads = 35
m = [0.1, 0.25, 0.5, 0.75, 0.9]
scale = 1.0
L = [6, 7, 8]


def apply_pca(data, n_components):
    if len(data.shape) == 3:
        data = data.unsqueeze(2)

    N, n_layers, n_heads, dim = data.shape
    principal_components = torch.zeros((n_layers, n_heads, n_components, dim), dtype=data.dtype, device=data.device)
    summed_variance = torch.zeros(n_layers, n_heads, dtype=data.dtype, device=data.device)

    # Loop over each of the (L, m) matrices
    for m in range(n_heads):
        for L in range(n_layers):
            matrix = data[:, L, m, :]
            mean_centered = matrix - matrix.mean(dim=0)
            cov_matrix = torch.mm(mean_centered.T, mean_centered) / (N - 1)
            eigenvalues, eigenvectors = LA.eigh(cov_matrix, UPLO='U')
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]
            principal_components[L, m, :, :] = eigenvectors[:, :n_components].T
            summed_variance[L, m] = eigenvalues[:n_components].sum() / eigenvalues.sum()

    return principal_components.cpu().data.numpy(), summed_variance.cpu().data.numpy()


def apply_svd(data, n_components):
    """
    Computes the SVD of a tensor and returns the first m column vectors of the U matrix.

    Args:
    tensor (torch.Tensor): The input tensor to compute the SVD on. It must be at least 2-D.
    m (int): The number of leading columns to return from the U matrix.

    Returns:
    torch.Tensor: The first m columns of the U matrix from the SVD of the input tensor.
    """
    if len(data.shape) == 3:
        data = data.unsqueeze(2)

    N, n_layers, n_heads, dim = data.shape
    column_vectors = torch.zeros((n_layers, n_heads, N, n_components), dtype=data.dtype, device=data.device)

    # Loop over each of the (L, m) matrices
    for m in range(n_heads):
        for L in range(n_layers):
            matrix = data[:, L, m, :]

            # Compute SVD
            U, S, V = torch.linalg.svd(matrix, full_matrices=False)
            column_vectors[L, m] = U[:, :n_components]

    # Return the first m columns of U
    return column_vectors, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LTV Inference: Learnable Task Vectors vs FVs on in-, out-of-dist data, and under dist shift')

    # Large batch sizes result in smoother curves
    parser.add_argument("--task", default="linear_regression", type=str, help='Task name', choices=['linear_regression', 'sparse_linear_regression', 'relu_2nn_regression'])
    parser.add_argument("--activation_type", default="attn", type=str, help='Type of the activation tracked', choices=['attn', 'hidden'])
    parser.add_argument("--vector_type", default="SVD", type=str, help='Type of the orthogonal vectors', choices=['PC', 'SVD'])
    parser.add_argument("--n_PCs", default=16, type=int, help='Number of principal components')

    # LTV training params
    parser.add_argument("--act_fn", default=None, type=str, help='Activation function at the LTV layer', choices=['leaky_relu', 'swish'])
    parser.add_argument("--ltv_seq_len", default=101, type=int, help='Sequence length LTV was trained on')

    # Logistics
    parser.add_argument("--seed", default=17, type=int, help='Seed number for PyTorch, NumPy and OpenAI Gym (default: 17)')
    parser.add_argument("--gpu", default=0, type=int, help='GPU ordinal for multi-GPU computers (default: 0)')

    args = parser.parse_args()
    args_dict = vars(parser.parse_args())

    # Obtain the task-specific experiment params
    task_name = args_dict['task']
    train_seq_len = 41 if "linear" in task_name else 101
    long_seq_len = 101 if "linear" in task_name else 201
    ltv_seq_lengths = [71, long_seq_len] if 'linear' in task_name else [151, long_seq_len]

    #  Set seeds
    set_seed(args_dict["seed"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    experiment_dir = f"./LTV_models/"

    task_dir = os.path.join(os.path.join(experiment_dir, task_name), str(args_dict["act_fn"]))

    # Prepare the model, get the task and data samplers, and save the parameters required later
    run_path = os.path.join(f"../models", task_name, "pretrained")

    model_dir = os.path.join(task_dir)
    load_dir = os.path.join(model_dir.replace("_models", "_ablation"), args_dict['activation_type'])
    os.makedirs(os.path.join(load_dir, args.vector_type), exist_ok=True)

    extraction_fn = apply_pca if args.vector_type == 'PC' else apply_svd

    # Load the attention activations
    # Transformer at the train length
    if args.ltv_seq_len == 101:
        attn_train = torch.load(os.path.join(load_dir, f'attn_out.pt')).to(device)
        attn_train_PCs, attn_train_PVE = extraction_fn(attn_train, args_dict['n_PCs'])
        attn_train_PCs = attn_train_PCs.squeeze() if args.activation_type == 'hidden' else attn_train_PCs
        torch.save(attn_train_PCs, os.path.join(load_dir, f'{args.vector_type}/attn_out.pt'))
        del attn_train
        del attn_train_PCs

        # Transformer at the max position
        attn_long = torch.load(os.path.join(load_dir, f'attn_out_max.pt')).to(device)
        attn_long_PCs, attn_long_PVE = extraction_fn(attn_long, args_dict['n_PCs'])
        attn_long_PCs = attn_long_PCs.squeeze() if args.activation_type == 'hidden' else attn_long_PCs
        torch.save(attn_long_PCs, os.path.join(load_dir, f'{args.vector_type}/attn_out_max.pt'))
        del attn_long
        del attn_long_PCs

        # Function Vector at the max position
        FV_attn_long = torch.load(os.path.join(load_dir, f'attn_out_FV.pt')).to(device)
        FV_attn_long_PCs, FV_attn_long_PVE = extraction_fn(FV_attn_long, args_dict['n_PCs'])
        FV_attn_long_PCs = FV_attn_long_PCs.squeeze() if args.activation_type == 'hidden' else FV_attn_long_PCs
        torch.save(FV_attn_long_PCs, os.path.join(load_dir, f'{args.vector_type}/attn_out_FV.pt'))
        del FV_attn_long
        del FV_attn_long_PCs

    # Learnable Task Vector trained at the max position at the max position
    LTV_attn_out_max = torch.load(os.path.join(load_dir, f'attn_out_LTV_{args.ltv_seq_len}.pt')).to(device)
    LTV_attn_out_max_PCs, LTV_attn_out_max_PVE = extraction_fn(LTV_attn_out_max, args_dict['n_PCs'])
    LTV_attn_out_max_PCs = LTV_attn_out_max_PCs.squeeze() if args.activation_type == 'hidden' else LTV_attn_out_max_PCs
    torch.save(LTV_attn_out_max_PCs, os.path.join(load_dir, f'{args.vector_type}/attn_out_LTV_{args.ltv_seq_len}.pt'))
    del LTV_attn_out_max
    del LTV_attn_out_max_PCs
