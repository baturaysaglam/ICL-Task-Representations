import argparse

from eval import gen_orthogonal_train_test
from samplers import GaussianSampler, sample_transformation
from tasks import NoisyLinearRegression

from utils.experiment import *
from utils.function_vector import *
from utils.learnable_task_vector import *
from utils.plot import *


n_FV_heads = 35
m = [0.1, 0.25, 0.5, 0.75, 0.9]
scale = 1.0
L = [6, 7, 8]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference of the trained Learnable Task Vectors')

    parser.add_argument("--dist_shift", default=None, type=str, help='Type of the distributional shift applied to the data', choices=['skewed', 'noisy_linear_regression'])

    # Large batch sizes result in smoother curves
    parser.add_argument("--batch_size", default=256, type=int, help='Batch size the results computed over')

    # LTV training params
    parser.add_argument("--act_fn", default=None, type=str, help='Activation function at the LTV layer', choices=['leaky_relu', 'swish'])

    # Logistics
    parser.add_argument("--seed", default=117, type=int, help='Seed number for PyTorch, NumPy and OpenAI Gym (default: 17)')
    parser.add_argument("--gpu", default=1, type=int, help='GPU ordinal for multi-GPU computers (default: 0)')

    args = parser.parse_args()
    args_dict = vars(parser.parse_args())

    # Obtain the task-specific experiment.sh params
    batch_size = args_dict['batch_size']

    #  Set seeds
    set_seed(args_dict["seed"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Construct the running and saving paths
    experiment_dir = f"./LTV_models/"
    tasks = os.listdir(experiment_dir)
    experiment_i, n_experiments = 0, len(tasks) * 6

    for task_name in tasks:
        if 'tree' in task_name:
            continue

        if args_dict['dist_shift'] == 'noisy_linear_regression' and task_name != 'linear_regression':
            continue

        task_dir = os.path.join(os.path.join(experiment_dir, task_name), str(args_dict["act_fn"]))
        seq_lens_dirs = os.listdir(task_dir)
        long_seq_len = 96 if "linear_regression" in task_name else 196

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

        # Compute the function vector
        with torch.no_grad():
            top_heads = top_heads_locations(indirect_effect_mat_np, n_FV_heads)
            universal_mean_activations = get_universal_mean_act(model, xs_sample, ys_sample, n_layers, n_heads, head_dim)
            FV = compute_function_vector(model, universal_mean_activations, top_heads, resid_dim, head_dim)

        ############################## LONG TASKS ##############################
        # Sample the data consisting of long prompts
        if args_dict['dist_shift'] == 'skewed':
            eigenvals = 1 / (torch.arange(n_dims) + 1)
            covar = sample_transformation(eigenvals, normalize=True)
            covariate_sampler = GaussianSampler(n_dims, bias=0, scale=covar)
            xs_test = covariate_sampler.sample_xs(b_size=batch_size, n_points=long_seq_len).to(device)
            ys_test = task.evaluate(xs_test).to(device)
        elif args_dict['dist_shift'] == 'noisy_linear_regression':
            if task_name != 'linear_regression':
                continue
            else:
                task = NoisyLinearRegression(n_dims, batch_size)
                xs_test = covariate_sampler.sample_xs(b_size=batch_size, n_points=long_seq_len).to(device)
                ys_test = task.evaluate(xs_test).to(device)
        elif args_dict['dist_shift'] == 'orthogonal_query':
            _, xs_test = gen_orthogonal_train_test(covariate_sampler, long_seq_len, batch_size)
            xs_test = xs_test.to(device)
            ys_test = task.evaluate(xs_test).to(device)
        else:
            xs_test = covariate_sampler.sample_xs(b_size=batch_size, n_points=long_seq_len).to(device)
            ys_test = task.evaluate(xs_test).to(device)

        with torch.no_grad():
            pred = evaluate_model(model, model, xs_test, ys_test, L=1, FV=None, dummy=None, scale=0.0)
            pred_FV = evaluate_model(model, model, xs_test, ys_test, L=L, FV=FV, dummy=m, scale=scale)

        for seq_len in seq_lens_dirs:
            if seq_len == 'None' or seq_len is None:
                continue

            seq_len_int = seq_len.split("_")[-1]
            ltv_model_dir = os.path.join(task_dir, seq_len)
            save_dir = ltv_model_dir.replace("_models", "_inference")
            save_dir = os.path.join(save_dir, args_dict['dist_shift']) if args_dict['dist_shift'] is not None else save_dir
            os.makedirs(save_dir, exist_ok=True)

            # Initialize the trained LTV layer
            params_path = os.path.join(ltv_model_dir, f"ltv_layer_{seq_len_int}.pth")
            ltv_layer = LearnableTaskVector(n_layers, n_heads).to(device)
            ltv_layer.load_state_dict(torch.load(params_path))
            ltv_layer.eval()

            with torch.no_grad():
                attn_out = get_attn_outs(model, xs_sample, ys_sample, n_layers, n_heads, head_dim, resid_dim)
                LTV = ltv_layer(attn_out)

            with torch.no_grad():
                pred_LTV = evaluate_model_on_LTV(model, model, xs_test, ys_test, LTV=LTV, dummy=None, scale=1.0)

                loss_dict = {"ys": ys_test.cpu().data.numpy(),
                             "transformer": pred.cpu().data.numpy(),
                             "transformer_FV": pred_FV.cpu().data.numpy(),
                             "transformer_LTV": pred_LTV.cpu().data.numpy()}
                np.savez(os.path.join(save_dir, "predictions.npz"), **loss_dict)
                experiment_i += 1
                print(f"Experiment {experiment_i} completed")
            ############################## LONG TASKS ##############################

