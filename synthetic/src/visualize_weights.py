import argparse

from utils.experiment import *
from utils.function_vector import *
from utils.learnable_task_vector import *
from utils.plot import *


n_FV_heads = 35
m = [0.1, 0.25, 0.5, 0.75, 0.9]
scale = 1.0
L = [6, 7, 8]


def normalize_matrix(matrix, axis=None):
  if axis is None:
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val) * 2 - 1
  else:
    return (matrix - matrix.min(axis=axis, keepdims=True)) / (matrix.max(axis=axis, keepdims=True) - matrix.min(axis=axis, keepdims=True)) * 2 - 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LTV Inference: Learnable Task Vectors vs FVs on in-, out-of-dist data, and under dist shift')

    # Large batch sizes result in smoother curves
    parser.add_argument("--batch_size", default=256, type=int, help='Batch size the results computed over')

    # LTV training params
    parser.add_argument("--act_fn", default=None, type=str, help='Activation function at the LTV layer', choices=['leaky_relu', 'swish'])

    # Logistics
    parser.add_argument("--seed", default=17, type=int, help='Seed number for PyTorch, NumPy and OpenAI Gym (default: 17)')
    parser.add_argument("--gpu", default=2, type=int, help='GPU ordinal for multi-GPU computers (default: 0)')

    args = parser.parse_args()
    args_dict = vars(parser.parse_args())

    # Obtain the task-specific experiment params
    batch_size = args_dict['batch_size']

    #  Set seeds
    set_seed(args_dict["seed"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Construct the running and saving paths for experiment 1: long prompts
    experiment_dir = f"./LTV_models/"
    tasks = os.listdir(experiment_dir)
    experiment_i, n_experiments = 0, len(tasks) * 6

    for task_name in ['relu_2nn_regression']:
        task_dir = os.path.join(os.path.join(experiment_dir, task_name), str(args_dict["act_fn"]))
        seq_lens_dirs = os.listdir(task_dir)
        seq_len_ints = [int(x.split('_')[-1]) for x in seq_lens_dirs]

        # Prepare the model, get the task and data samplers, and save the parameters required later
        run_path = os.path.join(f"../models", task_name, "pretrained")
        model, conf, task_sampler, covariate_sampler, params = prepare_model(run_path, batch_size, device)
        model.eval()
        n_dims, resid_dim, n_layers, n_heads, head_dim = params

        task = task_sampler()
        xs_sample = covariate_sampler.sample_xs(b_size=batch_size, n_points=conf.training.curriculum.points.end).to(device)
        ys_sample = task.evaluate(xs_sample).to(device)

        # Compute the indirect effects
        indirect_effect_mat = get_top_heads(model, xs_sample, ys_sample, n_layers, n_heads, verbose=False)
        indirect_effect_mat_np = indirect_effect_mat.cpu().data.numpy()
        # indirect_effect_mat_np = normalize_matrix(indirect_effect_mat_np)

        create_heatmap(indirect_effect_mat_np, x_label='Layer index', y_label='Head index', cbar_title='FV Indirect Effects ',
                       title=None, dpi=100, save_dir=None, show=True)

        for ltv_seq_len in seq_len_ints:
            model_dir = os.path.join(task_dir, f'seq_len_{ltv_seq_len}')

            # Initialize the trained LTV layer
            params_path = os.path.join(model_dir, f"ltv_layer_{ltv_seq_len}.pth")
            ltv_layer = LearnableTaskVector(n_layers, n_heads).to(device)
            ltv_layer.load_state_dict(torch.load(params_path))
            ltv_layer.eval()
            ltv_weights = ltv_layer.weights.cpu().data.numpy()
            ltv_weights = normalize_matrix(ltv_weights, axis=1)

            create_heatmap(ltv_weights.T, x_label='Layer index', y_label='Head index', cbar_title='LTV weights',
                           title=None, dpi=100, save_dir=None, show=True)
