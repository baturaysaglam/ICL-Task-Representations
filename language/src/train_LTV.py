import argparse
from collections import deque
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from LTV_utils.data import set_seed, sample_attn_out, sample_data, forward_pass
from LTV_utils.LTV import LearnableTaskVector
from FV_utils.model_utils import load_gpt_model_and_tokenizer
from FV_utils.prompt_utils import load_dataset


VOCAB_SIZE = 50400
VERBOSE_FREQ = 20


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Learnable Task Vectors on language tasks')

    # Task
    parser.add_argument("--dataset", default="mixed", choices=["antonym", "synonym", "mixed"])
    parser.add_argument("--model_name", default="EleutherAI/gpt-j-6b")

    # Large batch sizes result in smoother curves
    parser.add_argument("--n_examples", default=5, type=int, help='Sequence length to train on')
    parser.add_argument("--batch_size", default=100, type=int, help='Batch size the results computed over')
    parser.add_argument("--sgd_batch_size", default=32, type=int, help='Batch size the results computed over')

    # LTV training params
    parser.add_argument("--lr", default=5e-5, type=float, help='Learning rate')
    parser.add_argument("--act_fn", default=None, type=str, help='Activation function at the LTV layer', choices=['leaky_relu', 'swish'])
    parser.add_argument("--n_iter", default=100000, type=int, help='Number of epochs to LTV training')
    parser.add_argument("--early_stoppage_tolerance", default=20, type=int, help='Allowable number of epochs with no improvement in validation loss')

    # Logistics
    parser.add_argument("--seed", default=0, type=int, help='Seed number for PyTorch, NumPy and OpenAI Gym (default: 17)')
    parser.add_argument("--gpu", default=0, type=int, help='GPU ordinal for multi-GPU computers (default: 0)')

    args = parser.parse_args()
    args_dict = vars(parser.parse_args())

    # Obtain the experiment-specific params
    batch_size = args_dict['batch_size']
    sgd_batch_size = args_dict['sgd_batch_size']
    n_examples = args_dict['n_examples']

    n_iter = args_dict["n_iter"]
    early_stoppage_tolerance = args_dict["early_stoppage_tolerance"]
    lr = args_dict["lr"]
    act_fn = args_dict["act_fn"]

    set_seed(args_dict["seed"])
    device = torch.device(f"cuda:{args.gpu}")

    model_name = args_dict["model_name"]
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)

    # Disable gradient updates for all transformer parameters
    for param in model.parameters():
        param.requires_grad = False

    # Construct the experiment dir
    experiment_dir = f"./LTV_models/{args.dataset}/{act_fn}/seq_len_{n_examples}"

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    n_layers = model_config['n_layers']
    resid_dim = model_config['resid_dim']
    n_heads = model_config['n_heads']
    head_dim = resid_dim // n_heads
    device = model.device

    ltv_layer = LearnableTaskVector(n_layers, n_heads, head_dim).to(device)
    loss_fn = F.cross_entropy
    optimizer = optim.Adam(ltv_layer.parameters(), lr=lr)

    if os.path.exists(os.path.join(experiment_dir, f"ltv_layer_{n_examples}.pth")):
        last_iter = 0
        for file_name in os.listdir(experiment_dir):
            if '.pth' in file_name:
                iter_num = int(file_name.split('.pth')[0].split('_')[-1])
                if iter_num > last_iter:
                    last_iter = iter_num
        if os.path.exists(os.path.join(experiment_dir, f"ltv_layer_{n_examples}_{last_iter}.pth")):
            checkpt_dir = os.path.join(experiment_dir, f"ltv_layer_{n_examples}_{last_iter}.pth")
        else:
            checkpt_dir = os.path.join(experiment_dir, f"ltv_layer_{n_examples}.pth")
        ltv_params = torch.load(checkpt_dir)
        ltv_layer.load_state_dict(ltv_params)
        with open(os.path.join(experiment_dir, 'min_val_loss.txt'), 'r') as file:
            lowest_val_loss = float(file.read())
        print("Existing training found - parameters loaded")
    else:
        lowest_val_loss = float('inf')  # init the lowest validation loss

    if args_dict["dataset"] != "mixed":
        dataset = [load_dataset(args_dict["dataset"], seed=args_dict["seed"])]
    else:
        dataset_1 = load_dataset("antonym", seed=args_dict["seed"])
        dataset_2 = load_dataset("synonym", seed=args_dict["seed"])
        dataset = (dataset_1, dataset_2)

    loss_verbose = deque(maxlen=VERBOSE_FREQ)
    loss_verbose_clean_logits = deque(maxlen=VERBOSE_FREQ)

    ltv_layer.train()  # Set the model to training mode

    for iter_i in range(n_iter):
        total_loss = 0.0

        attn_out = sample_attn_out(dataset, model, model_config, tokenizer, batch_size)
        lt_vector = ltv_layer.forward(attn_out)
        sentences, targets = sample_data(dataset, n_examples, sgd_batch_size, shuffle_labels=True)
        logits, clean_logits, target_indices = forward_pass(model, model_config, tokenizer, VOCAB_SIZE, sentences, targets, lt_vector)

        loss = loss_fn(logits, target_indices.to(torch.int64))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_verbose.append(loss.item())
        loss_verbose_clean_logits.append(loss_fn(clean_logits, target_indices.to(torch.int64)).detach().item())

        # Clean up some space
        del lt_vector
        del attn_out
        del logits
        del clean_logits
        del target_indices

        if iter_i % VERBOSE_FREQ == 0:
            print(f'Epoch [{iter_i + 1}/{n_iter}] - training loss: {np.mean(loss_verbose):.4f}, clean logits: {np.mean(loss_verbose_clean_logits):.4f}')

        if loss.item() < lowest_val_loss:
            lowest_val_loss = loss.item()
            torch.save(ltv_layer.state_dict(), os.path.join(experiment_dir, f'ltv_layer_{n_examples}.pth'))
            with open(os.path.join(experiment_dir, 'min_val_loss.txt'), 'w') as file:
                file.write(str(lowest_val_loss))

            print(f'Checkpoint saved - iter: {iter_i + 1}, loss: {loss.item():.4f}')

        if iter_i % 200 == 0:
            torch.save(ltv_layer.state_dict(), os.path.join(experiment_dir, f'ltv_layer_{n_examples}_{iter_i}.pth'))
