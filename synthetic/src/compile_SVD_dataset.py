import argparse

from utils.experiment import *
from utils.function_vector import *
from utils.learnable_task_vector import *


n_FV_heads = 35
m = [0.1, 0.25, 0.5, 0.75, 0.9]
scale = 1.0
L = [6, 7, 8]


def calculate_memory_allocation(tensor):
    num_elements = tensor.numel()
    element_size = tensor.element_size()
    memory_allocation_bytes = num_elements * element_size
    memory_allocation_mb = memory_allocation_bytes / (1024 ** 2)
    return memory_allocation_mb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a dataset for SVD by collecting attention and layer activations')

    # Large batch sizes result in smoother curves
    parser.add_argument("--algo", default="LTV", type=str, help='Approach name', choices=['vanilla', 'vanilla_max', 'FV', 'LTV'])
    parser.add_argument("--task", default="linear_regression", type=str, help='Task name', choices=['linear_regression', 'sparse_linear_regression', 'relu_2nn_regression'])
    parser.add_argument("--batch_size", default=256, type=int, help='Batch size the results computed over')

    parser.add_argument("--size", default=100, type=int, help='Dataset size as the number of batches')

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
    batch_size = args_dict['batch_size']

    #  Set seeds
    set_seed(args_dict["seed"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    experiment_dir = f"./LTV_models/"

    task_dir = os.path.join(os.path.join(experiment_dir, task_name), str(args_dict["act_fn"]))

    # Prepare the model, get the task and data samplers, and save the parameters required later
    run_path = os.path.join(f"../models", task_name, "pretrained")
    model, conf, task_sampler, covariate_sampler, params = prepare_model(run_path, batch_size, device)
    model.eval()
    n_dims, resid_dim, n_layers, n_heads, head_dim = params

    # Freeze all transformer parameters
    for param in model.parameters():
        param.requires_grad = False

    train_seq_len = conf.training.curriculum.points.end
    ltv_seq_len = args_dict['ltv_seq_len']

    if args_dict['algo'] == 'vanilla':
        test_seq_len = train_seq_len - 2
        mid_len = None
    else:
        test_seq_len = 101 if "linear_regression" in task_name else 201
        mid_len = 56 if "linear_regression" in task_name else 126

    model_dir = os.path.join(task_dir)
    save_dir = model_dir.replace("_models", "_ablation")
    os.makedirs(save_dir, exist_ok=True)

    if args_dict['algo'] == 'vanilla':
        save_name = 'attn_out'
    elif args_dict['algo'] == 'vanilla_max':
        save_name = 'attn_out_max'
    elif args_dict['algo'] == 'FV':
        save_name = 'attn_out_FV'
    elif args_dict['algo'] == 'LTV':
        save_name = f'attn_out_LTV_{ltv_seq_len}'

    task = task_sampler()

    if args_dict['algo'] == 'FV' or 'LTV' in args_dict['algo']:
        xs_sample = covariate_sampler.sample_xs(b_size=batch_size, n_points=train_seq_len).to(device)
        ys_sample = task.evaluate(xs_sample).to(device)

        if args_dict['algo'] == 'FV':
            # Function Vectors
            indirect_effect_mat = get_top_heads(model, xs_sample, ys_sample, n_layers, n_heads)
            indirect_effect_mat_np = indirect_effect_mat.cpu().data.numpy()

            n_FV_heads = 35

            with torch.no_grad():
                top_heads = top_heads_locations(indirect_effect_mat_np, n_FV_heads)
                universal_mean_activations = get_universal_mean_act(model, xs_sample, ys_sample, n_layers, n_heads, head_dim)
                FV = compute_function_vector(model, universal_mean_activations, top_heads, resid_dim, head_dim)

                L = [6, 7, 8]
                m = [0.1, 0.25, 0.5, 0.75, 0.9]
                scale = 1.0
        else:
            params_path = os.path.join(model_dir, f"seq_len_{ltv_seq_len}/ltv_layer_{ltv_seq_len}.pth")
            ltv_layer = LearnableTaskVector(n_layers, n_heads).to(device)
            ltv_layer.load_state_dict(torch.load(params_path))
            ltv_layer.eval()

            # Freeze all transformer parameters
            for param in ltv_layer.parameters():
                param.requires_grad = False

            with torch.no_grad():
                attn_out = get_attn_outs(model, xs_sample, ys_sample, n_layers, n_heads, head_dim, resid_dim)
                LTV = ltv_layer(attn_out)

    # Initialize the attention set
    # attn_activations = torch.zeros((int(args_dict['size'] * args_dict['batch_size']), n_layers, n_heads, resid_dim))
    layer_activations = torch.zeros((int(args_dict['size'] * args_dict['batch_size']), n_layers + 1, resid_dim))

    for iter_i in range(args_dict['size']):
        # First sample the long data
        xs_long = covariate_sampler.sample_xs(b_size=batch_size, n_points=test_seq_len).to(device)
        ys_long = task.evaluate(xs_long).to(device)

        with torch.no_grad():
            if 'vanilla' in args_dict['algo']:
                attn_out, hidden_state = get_activations(model, xs_long, ys_long, n_layers, n_heads, head_dim, resid_dim, type=None, LTV=None, FV=None, FV_layers=None, dummy_indices=None, scale=0.0)
            else:
                if args_dict['algo'] == 'FV':
                    attn_out, hidden_state = get_activations(model, xs_long, ys_long, n_layers, n_heads, head_dim, resid_dim, type='FV', LTV=None, FV=FV, FV_layers=L, dummy_indices=m, scale=scale)
                else:
                    attn_out, hidden_state = get_activations(model, xs_long, ys_long, n_layers, n_heads, head_dim, resid_dim, type='LTV', LTV=LTV, FV=None, FV_layers=None, dummy_indices=None, scale=1.0)

        # attn_activations[iter_i * args_dict['batch_size']:(iter_i + 1) * args_dict['batch_size']] = attn_out.cpu()
        layer_activations[iter_i * args_dict['batch_size']:(iter_i + 1) * args_dict['batch_size']] = hidden_state.cpu()
        print(f'{args.algo} - Iter {iter_i + 1} completed')

    attn_path = os.path.join(save_dir, 'attn')
    hidden_path = os.path.join(save_dir, 'hidden')

    os.makedirs(attn_path, exist_ok=True)
    os.makedirs(hidden_path, exist_ok=True)

    # torch.save(attn_activations, os.path.join(attn_path, f'{save_name}.pt'))
    torch.save(layer_activations, os.path.join(hidden_path, f'{save_name}.pt'))
