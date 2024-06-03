import argparse
import sys
import time

import seaborn as sns

from utils.experiment import *
from utils.function_vector import *
from utils.learnable_task_vector import *
from utils.plot import *


sys.path.append("..")  # Adds higher directory to python modules path

# sns.set_theme('notebook', 'darkgrid')
# palette = sns.color_palette('colorblind')

sns.set(context='paper', style='ticks', palette='colorblind')

NORM_CURVE = False
COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e']


# Normalizing function
def normalize(lst, min_val, max_val):
    return [(x - min_val) / (max_val - min_val) for x in lst]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the loss curves based on the saved model predictions')

    parser.add_argument("--dist_shift", default='orthogonal_query', type=str,
                        help='Type of the distributional shift applied to the data',
                        choices=['skewed', 'noisy_linear_regression'])

    # LTV training params
    parser.add_argument("--act_fn", default=None, type=str, help='Activation function at the LTV layer', choices=['leaky_relu', 'swish'])

    # Logistics
    parser.add_argument("--seed", default=17, type=int, help='Seed number for PyTorch, NumPy and OpenAI Gym (default: 17)')
    parser.add_argument("--gpu", default=2, type=int, help='GPU ordinal for multi-GPU computers (default: 0)')

    args = parser.parse_args()
    args_dict = vars(parser.parse_args())

    #  Set seeds
    set_seed(args_dict["seed"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Construct the running and saving paths for experiment 1: long prompts

    experiment_dir = f"./LTV_inference/"
    tasks = os.listdir(experiment_dir)
    experiment_i, n_experiments = 0, len(tasks) * 6

    for task_name in tasks:
        task_dir = os.path.join(os.path.join(experiment_dir, task_name), str(args_dict["act_fn"]))

        if 'figures' in task_name:
            continue

        seq_lens_dirs = os.listdir(task_dir)

        long_seq_len = 96 if "linear_regression" in task_name else 196

        if 'tree' in task_name:
            continue

        if args_dict['dist_shift'] == 'noisy_linear_regression' and 'linear_regression' != task_name:
            continue

        for seq_len in seq_lens_dirs:
            if seq_len == 'None' or seq_len is None:
                continue

            result_dir = os.path.join(task_dir, seq_len) if args_dict['dist_shift'] is None else os.path.join(task_dir, seq_len, args_dict['dist_shift'])
            seq_len_int = seq_len.split("_")[-1]
            predictions = []

            for file_name in os.listdir(result_dir):
                if "npz" not in file_name:
                    continue

                save_dir = os.path.join(result_dir, file_name)
                predictions = np.load(save_dir, allow_pickle=True)

                ys, pred, pred_FV, pred_LTV = predictions["ys"], predictions["transformer"], predictions["transformer_FV"], predictions["transformer_LTV"]

                loss, conf_int = distance(pred, ys), compute_confidence_interval((pred - ys) ** 2)
                loss_FV, conf_int_FV = distance(pred_FV, ys), compute_confidence_interval((pred_FV - ys) ** 2)
                loss_LTV, conf_int_LTV = distance(pred_LTV, ys), compute_confidence_interval((pred_LTV - ys) ** 2)

                if NORM_CURVE:
                    max_value = max(max(loss), max(loss_FV), max(loss_LTV))
                    loss = normalize(loss, 0.0, max_value)
                    loss_FV = normalize(loss_FV, 0.0, max_value)
                    loss_LTV = normalize(loss_LTV, 0.0, max_value)

                    y_ticks_max = 1.0
                    y_ticks_interval = 0.25
                else:
                    y_ticks_max = 20.0
                    y_ticks_interval = 2.5

                losses = [loss, loss_FV, loss_LTV]
                legends = ["Transformer", r"Transformer + $v$", r"Transformer + $v_\theta$"]

                if "long" in file_name:
                    fig_save_name = "long_task.png"
                    title = r"$\bf{longer prompts:}$ $\mathcal{N}(0, 1)$"
                elif "out" in file_name:
                    fig_save_name = "out_of_dist.png"
                    title = r"$\bf{out-of-dist:}$ $\mathcal{N}(0, 1)$ $\rightarrow$ $U(-2, 2)$"
                else:
                    fig_save_name = "dist_shift.png"
                    title = r"$\bf{dist}$ $\bf{shift:}$ $\mathcal{N}(0, 1)$ $\rightarrow$ $\mathcal{N}(-0.75, 1.25)$"

                if args_dict['dist_shift'] is not None:
                    dist_shift_save_name = os.path.join(experiment_dir, 'figures', task_name, f'{args.dist_shift}_{seq_len_int}.pdf')
                    os.makedirs(os.path.join(experiment_dir, 'figures', task_name), exist_ok=True)

                title = r"$\bf{L:}$" + f" {seq_len_int} " + title

                time.sleep(0.1)

                plot_transformer(losses,
                                 legends,
                                 title=None,
                                 colors=COLORS,
                                 ci_widths=[conf_int, conf_int_FV, conf_int_LTV],
                                 x_label="# in-context examples",
                                 y_label="Mean squared error",
                                 baseline=None,
                                 save_path=os.path.join(result_dir, fig_save_name) if args_dict['dist_shift'] is None else dist_shift_save_name,
                                 y_ticks_max=3 if args_dict['dist_shift'] == "orthogonal_query" else y_ticks_max,
                                 y_ticks_interval=1 if args_dict['dist_shift'] == "orthogonal_query" else y_ticks_interval,
                                 dpi=150,
                                 show=True)
