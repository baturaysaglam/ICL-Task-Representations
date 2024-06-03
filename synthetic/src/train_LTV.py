import argparse
import sys

import seaborn as sns
import torch.optim as optim
import torch.nn.functional as F

from utils.experiment import *
from utils.learnable_task_vector import *
from torch.utils.data import TensorDataset, DataLoader, random_split
from utils.function_vector import *
from utils.plot import *


sys.path.append("..")  # Adds higher directory to python modules path

sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Learnable Task Vectors on synthetic tasks')

    # Task
    parser.add_argument("--task", default="sparse_linear_regression", help='Task for transformer to perform through ICL',
                        choices=['linear_regression', 'sparse_linear_regression', 'decision_tree', 'relu_2nn_regression'])

    # Large batch sizes result in smoother curves
    parser.add_argument("--seq_len", default=56, type=int, help='Sequence length to train on')
    parser.add_argument("--batch_size", default=256, type=int, help='Batch size the results computed over')

    # LTV training params
    parser.add_argument("--lr", default=2.5e-5, type=float, help='Learning rate')
    parser.add_argument("--act_fn", default=None, type=str, help='Activation function at the LTV layer', choices=['leaky_relu', 'swish'])
    parser.add_argument("--sgd_batch_size", default=256, type=int, help='Batch size in training the LTV')
    parser.add_argument("--n_epochs", default=10000, type=int, help='Number of epochs to LTV training')
    parser.add_argument("--early_stoppage_tolerance", default=7, type=int, help='Allowable number of epochs with no improvement in validation loss')
    parser.add_argument("--n_batches", default=100, type=int, help='Total num of samples: batch_size x num_batches')

    # Logistics
    parser.add_argument("--seed", default=17, type=int, help='Seed number for PyTorch, NumPy and OpenAI Gym (default: 17)')
    parser.add_argument("--gpu", default=2, type=int, help='GPU ordinal for multi-GPU computers (default: 0)')

    args = parser.parse_args()
    args_dict = vars(parser.parse_args())

    # Obtain the task-specific experiment params
    task_name = args_dict['task']

    batch_size = args_dict['batch_size']
    seq_len = args_dict['seq_len']

    sgd_batch_size = args_dict["sgd_batch_size"]
    n_epochs = args_dict["n_epochs"]
    early_stoppage_tolerance = args_dict["early_stoppage_tolerance"]
    n_batches = args_dict["n_batches"]
    lr = args_dict["lr"]
    act_fn = args_dict["act_fn"]

    # Make sure to delete these since we don't treat them as a parameter
    del args_dict['gpu']

    #  Set seeds
    set_seed(args_dict["seed"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Construct the running and saving paths
    figures_path, preds_path = prepare_save_dirs(os.path.join(task_name, "LTV_models"))
    run_path = os.path.join(f"../models", task_name, "pretrained")

    # Prepare the model, get the task and data samplers, and save the parameters required later
    model, conf, task_sampler, covariate_sampler, params = prepare_model(run_path, batch_size, device)
    n_dims, resid_dim, n_layers, n_heads, head_dim = params

    # Freeze all transformer parameters
    for param in model.parameters():
        param.requires_grad = False

    # Construct the experiment dir
    experiment_dir = f"./LTV_models/{task_name}/{act_fn}/seq_len_{seq_len}"

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    task_long = task_sampler()
    task_long_w = task_long.w_b.squeeze().to(device) if "linear" in task_name else None
    xs_long = covariate_sampler.sample_xs(b_size=batch_size, n_points=seq_len).to(device)
    ys_long = task_long.evaluate(xs_long).to(device)

    n_samples = int(n_batches * batch_size)

    if os.path.exists(os.path.join(experiment_dir, 'data_samples.pt')) and os.path.exists(
            os.path.join(experiment_dir, 'long_data.pt')):
        data_samples = torch.load(os.path.join(experiment_dir, 'data_samples.pt'))
        label_samples = torch.load(os.path.join(experiment_dir, 'label_samples.pt'))
        long_data = torch.load(os.path.join(experiment_dir, 'long_data.pt'))
        long_label = torch.load(os.path.join(experiment_dir, 'long_label.pt'))

        print("Data loaded")
    else:
        data_samples = torch.zeros(n_samples, n_layers, n_heads, resid_dim)
        label_samples = torch.zeros(n_samples, 1, dtype=model._backbone.dtype)
        long_data = torch.zeros(n_samples, seq_len, n_dims)
        long_label = torch.zeros(n_samples, seq_len, dtype=model._backbone.dtype)

        curr_idx, batch_i = 0, 0

        while curr_idx < n_samples:
            data, labels, xs_long, ys_long = data_sampler(model, conf, task_sampler, covariate_sampler, seq_len,
                                                          batch_size, n_layers, n_heads, head_dim, resid_dim, device)
            labels = labels[:, -1].unsqueeze(dim=1)

            space_left = n_samples - curr_idx
            end_idx = curr_idx + min(batch_size, space_left)

            data_samples[curr_idx:end_idx] = data[:end_idx - curr_idx]
            label_samples[curr_idx:end_idx] = labels[:end_idx - curr_idx]
            long_data[curr_idx:end_idx] = xs_long[:end_idx - curr_idx]
            long_label[curr_idx:end_idx] = ys_long[:end_idx - curr_idx]

            curr_idx = end_idx
            batch_i += 1

            print(f"Added batch {batch_i}")

        data_samples = data_samples[:curr_idx]
        label_samples = label_samples[:curr_idx]
        long_data = long_data[:curr_idx]
        long_label = long_label[:curr_idx]

        torch.save(data_samples, os.path.join(experiment_dir, 'data_samples.pt'))
        torch.save(label_samples, os.path.join(experiment_dir, 'label_samples.pt'))
        torch.save(long_data, os.path.join(experiment_dir, 'long_data.pt'))
        torch.save(long_label, os.path.join(experiment_dir, 'long_label.pt'))

        print("Data saved")

    # Split dataset into training and validation sets
    dataset = TensorDataset(data_samples, label_samples, long_data, long_label)
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=sgd_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=sgd_batch_size, shuffle=True)

    verbose_freq = 1

    ltv_layer = LearnableTaskVector(n_layers, n_heads).to(device)
    loss_fn = F.huber_loss
    optimizer = optim.Adam(ltv_layer.parameters(), lr=lr)

    lowest_val_loss = float('inf')  # init lowest validation loss
    no_improvement_count = 0  # count epochs with no improvement in validation loss

    for epoch_i in range(n_epochs):
        ltv_layer.train()  # Set the model to training mode
        total_loss = 0.0
        num_batches = 0

        # Training phase
        for batch_i, (inputs, ys, xs_long, ys_long) in enumerate(train_dataloader):
            inputs, ys = inputs.to(device), ys.to(device)
            xs_long, ys_long = xs_long.to(device), ys_long.to(device)

            optimizer.zero_grad()

            ltv_out = ltv_layer(inputs)
            ys_pred = add_learnable_task_vector(model, xs_long, ys_long, ltv_out, dummy_indices=None, scale=1.0)
            ys_pred = ys_pred[:, -1].unsqueeze(dim=1)

            loss = loss_fn(ys_pred, ys)
            total_loss += loss.item()
            num_batches += 1

            loss.backward()
            optimizer.step()

        average_training_loss = total_loss / num_batches

        # Validation phase
        ltv_layer.eval()  # Set the model to evaluation mode
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for inputs, ys, xs_long, ys_long in val_dataloader:
                inputs, ys = inputs.to(device), ys.to(device)
                xs_long, ys_long = xs_long.to(device), ys_long.to(device)

                ltv_out = ltv_layer(inputs)
                ys_pred = add_learnable_task_vector(model, xs_long, ys_long, ltv_out, dummy_indices=None, scale=1.0)
                ys_pred = ys_pred[:, -1].unsqueeze(dim=1)

                val_loss = loss_fn(ys_pred, ys)
                total_val_loss += val_loss.item()
                num_val_batches += 1

        average_val_loss = total_val_loss / num_val_batches
        print(
            f'Epoch [{epoch_i + 1}/{n_epochs}] - training loss: {average_training_loss:.4f}, validation loss: {average_val_loss:.4f}')

        # Save model if validation loss is the lowest
        if average_val_loss < lowest_val_loss:
            lowest_val_loss = average_val_loss
            torch.save(ltv_layer.state_dict(), os.path.join(experiment_dir, f'ltv_layer_{seq_len}.pth'))
            print('Checkpoint saved with lowest validation loss')
        else:
            no_improvement_count += 1

        # Early stopping check
        if no_improvement_count >= early_stoppage_tolerance:
            print(
                f'Early stopping triggered after {no_improvement_count} epochs without improvement in validation loss.')
            break
