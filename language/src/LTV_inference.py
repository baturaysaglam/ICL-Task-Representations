import argparse
import os, re, json
from scipy import stats
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
torch.set_grad_enabled(False)

from src.utils.extract_utils import get_mean_head_activations, compute_universal_function_vector
from src.utils.intervention_utils import fv_intervention_natural_text, function_vector_intervention
from src.utils.model_utils import load_gpt_model_and_tokenizer
from src.utils.prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt
from src.utils.eval_utils import decode_to_vocab, sentence_eval

from src.ltv_utils.data import set_seed, sample_data, forward_pass
from src.ltv_utils.LTV import LearnableTaskVector
from src.ltv_utils.extract_utils import get_attn_out, get_head_activations_on_prompt
from src.ltv_utils.intervention_utils import ltv_intervention, ltv_intervention_natural_text


VOCAB_SIZE = 50400
FV_EDIT_LAYER = 9
N_EXAMPLES = 10


def compute_perplexity(logits, target_index):
    # logits: Tensor of shape [batch_size, num_tokens]
    # target_index: Tensor of shape [batch_size] containing indices of the correct words
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_target = log_probs.gather(dim=-1, index=target_index.unsqueeze(-1)).squeeze(-1)

    avg_neg_log_prob = -log_probs_target.mean()

    perplexity = torch.exp(avg_neg_log_prob)

    return perplexity.item()


def compute_loss(loss_fn, logits, target_index):
    return loss_fn(logits, target_index).detach().item()


def mean_and_confidence_interval(data):
    """Calculate the mean and 95% confidence interval of a given list of numbers."""
    data = np.array(data)

    mean = np.mean(data)
    sem = stats.sem(data)

    # Determine the t-critical value for 95% confidence
    # Use the t-distribution's critical value (two-tailed, 95%)
    confidence_level = 0.95
    degrees_of_freedom = len(data) - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

    # Calculate the margin of error
    margin_of_error = t_critical * sem

    # Calculate the confidence interval
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    return mean, confidence_interval, margin_of_error


def divide_dict_values(data, divisor):
    """ Recursively divide all numerical values in the dictionary by 'divisor'. """
    for key, value in data.items():
        if isinstance(value, dict):
            # If the value is another dictionary, recurse into it
            divide_dict_values(value, divisor)
        else:
            # Otherwise, perform the division, ensuring the value is numeric
            if isinstance(value, (int, float)):
                data[key] = value / divisor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Experiment 0: Learnable Task Vectors vs FVs on in-, out-of-dist data, and under dist shift')

    # Task
    parser.add_argument("--task_names", type=str, nargs='*', default=['antonym', 'synonym'], help='Task names')
    parser.add_argument("--prompt_type", type=str, default='few-shot', help='Type of prompting')
    parser.add_argument("--model_name", default="EleutherAI/gpt-j-6b")

    # Large batch sizes result in smoother curves
    parser.add_argument("--n_trials", default=256, type=int, help='Number of seeds')
    parser.add_argument("--batch_size", default=100, type=int, help='Batch size the results computed over')

    # Logistics
    parser.add_argument("--start_seed", default=17, type=int, help='Seed number for PyTorch, NumPy and OpenAI Gym (default: 17)')
    parser.add_argument("--gpu", default=0, type=int, help='GPU ordinal for multi-GPU computers (default: 0)')

    args = parser.parse_args()
    args_dict = vars(parser.parse_args())

    # Obtain the experiment-specific params
    task_names = args_dict['task_names']
    n_trials = args_dict['n_trials']

    device = torch.device("cuda:" + str(args_dict['gpu']) if torch.cuda.is_available() else "cpu")

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args_dict['model_name'], device=device)

    n_layers = model_config['n_layers']
    resid_dim = model_config['resid_dim']
    n_heads = model_config['n_heads']
    head_dim = resid_dim // n_heads

    act_fn = None
    loss_fn = F.cross_entropy
    lt_seq_len = 5

    vanilla_perf, FV_perf, LTV_perf, mixed_LTV_perf = {}, {}, {}, {}

    for task_name in task_names:
        vanilla_perf[task_name] = {'filtered': {'perplexity': [], 'accuracy': [], 'loss': []}, 'unfiltered': {'perplexity': [], 'accuracy': [], 'loss': []}}
        FV_perf[task_name] = {'filtered': {'perplexity': [], 'accuracy': [], 'loss': []}, 'unfiltered': {'perplexity': [], 'accuracy': [], 'loss': []}}
        LTV_perf[task_name] = {'filtered': {'perplexity': [], 'accuracy': [], 'loss': []}, 'unfiltered': {'perplexity': [], 'accuracy': [], 'loss': []}}
        mixed_LTV_perf[task_name] = {'filtered': {'perplexity': [], 'accuracy': [], 'loss': []}, 'unfiltered': {'perplexity': [], 'accuracy': [], 'loss': []}}

        current_directory = os.getcwd()
        path_to_model = os.path.join(os.path.dirname(current_directory), f"src/LTV_models/{task_name}/{act_fn}/seq_len_{lt_seq_len}")
        path_to_model = os.path.join(path_to_model, f"ltv_layer_{lt_seq_len}.pth")
        path_to_mixed_model = path_to_model.replace(task_name, 'mixed')

        ltv_layer = LearnableTaskVector(n_layers, n_heads, head_dim).to(device)
        ltv_params = torch.load(path_to_model)
        ltv_layer.load_state_dict(ltv_params)

        mixed_ltv_layer = LearnableTaskVector(n_layers, n_heads, head_dim).to(device)
        mixed_ltv_params = torch.load(path_to_mixed_model)
        mixed_ltv_layer.load_state_dict(mixed_ltv_params)

        trials = 0

        while trials < args_dict['n_trials']:
            dataset = load_dataset(task_name, seed=args_dict['start_seed'] + trials)
            mean_activations, _ = get_mean_head_activations(dataset, model, model_config, tokenizer)

            # Compute FV and LTV
            FV, top_heads = compute_universal_function_vector(mean_activations, model, model_config, n_top_heads=10)

            with torch.no_grad():
                attn_out = get_attn_out(mean_activations, model, model_config)
                lt_vector = ltv_layer.forward(attn_out)
                lt_vector = lt_vector.squeeze()

                mixed_lt_vector = mixed_ltv_layer.forward(attn_out)
                mixed_lt_vector = mixed_lt_vector.squeeze()

            # Sample ICL example pairs, and a test word
            test_idx = np.random.randint(0, len(dataset['test']))

            word_pairs = dataset['train'][np.random.choice(len(dataset['train']), N_EXAMPLES, replace=False)]
            test_pair = dataset['test'][test_idx]

            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=test_pair, prepend_bos_token=True)
            sentence = create_prompt(prompt_data)

            if args_dict['prompt_type'] == 'few-shot':
                test_prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=test_pair, prepend_bos_token=True, shuffle_labels=True)
                test_sentence = create_prompt(test_prompt_data)
            elif args_dict['prompt_type'] == 'zero-shot':
                test_prompt_data = word_pairs_to_prompt_data({'input': [], 'output': []}, query_target_pair=test_pair, prepend_bos_token=True, shuffle_labels=True)
                test_sentence = create_prompt(test_prompt_data)
            else:
                raise NotImplementedError('No other prompting type is available')

            # Intervention on the few-shot prompt
            clean_logits, fv_logits = function_vector_intervention(test_sentence, [test_pair['output']], FV_EDIT_LAYER, FV, model, model_config, tokenizer)
            _, ltv_logits = ltv_intervention(test_sentence, [test_pair['output']], lt_vector, model, model_config, tokenizer)
            _, mixed_ltv_logits = ltv_intervention(test_sentence, [test_pair['output']], mixed_lt_vector, model, model_config, tokenizer)

            # Track the predictions
            target_idx = tokenizer.encode(" " + test_pair['output'])
            target_idx = torch.tensor(target_idx, dtype=torch.int64).to(device)

            try:
                vanilla_pred = decode_to_vocab(clean_logits, tokenizer, k=1)[0][0].split(' ')[-1]
                FV_pred = decode_to_vocab(fv_logits, tokenizer, k=1)[0][0].split(' ')[-1]
                LTV_pred = decode_to_vocab(ltv_logits, tokenizer, k=1)[0][0].split(' ')[-1]
                mixed_LTV_pred = decode_to_vocab(mixed_ltv_logits, tokenizer, k=1)[0][0].split(' ')[-1]

                if test_pair['output'] != FV_pred and test_pair['output'] != LTV_pred and test_pair['output'] != vanilla_pred:
                    increment_trials = False
                    filter_keys = ['unfiltered']
                else:
                    increment_trials = True
                    filter_keys = ['unfiltered', 'filtered']

                for filter in filter_keys:
                    vanilla_perf[task_name][filter]['perplexity'].append(compute_perplexity(clean_logits, target_idx))
                    FV_perf[task_name][filter]['perplexity'].append(compute_perplexity(fv_logits, target_idx))
                    LTV_perf[task_name][filter]['perplexity'].append(compute_perplexity(ltv_logits, target_idx))
                    mixed_LTV_perf[task_name][filter]['perplexity'].append(compute_perplexity(mixed_ltv_logits, target_idx))

                    vanilla_perf[task_name][filter]['loss'].append(compute_loss(loss_fn, clean_logits, target_idx))
                    FV_perf[task_name][filter]['loss'].append(compute_loss(loss_fn, fv_logits, target_idx))
                    LTV_perf[task_name][filter]['loss'].append(compute_loss(loss_fn, ltv_logits, target_idx))
                    mixed_LTV_perf[task_name][filter]['loss'].append(compute_loss(loss_fn, mixed_ltv_logits, target_idx))

                    vanilla_perf[task_name][filter]['accuracy'].append(vanilla_pred == test_pair['output'])
                    FV_perf[task_name][filter]['accuracy'].append(FV_pred == test_pair['output'])
                    LTV_perf[task_name][filter]['accuracy'].append(LTV_pred == test_pair['output'])
                    mixed_LTV_perf[task_name][filter]['accuracy'].append(mixed_LTV_pred == test_pair['output'])
            except RuntimeError:
                # print(f'Exception occurred')
                continue

            trials += 1 if increment_trials else 0
            # print(f'For {task_name}, seed {trials} completed')

    divide_dict_values(vanilla_perf, args_dict['n_trials'])
    divide_dict_values(FV_perf, args_dict['n_trials'])
    divide_dict_values(LTV_perf, args_dict['n_trials'])
    divide_dict_values(mixed_LTV_perf, args_dict['n_trials'])

    model_names = ['Vanilla transformer', 'Transformer + FV', 'Transformer + LTV', 'Transformer + LTV (mixed)']
    results = [vanilla_perf, FV_perf, LTV_perf, mixed_LTV_perf]

    print(f'\n\n\n---------- {args.prompt_type.capitalize()} Results ----------')
    for filter_key in ['unfiltered', 'filtered']:
        print(f'\n---------- {filter_key.capitalize()} ----------')
        for task_name in task_names:
            for i, (model_result, model_name) in enumerate(zip(results, model_names)):
                accuracies, perplexities, losses = np.zeros(n_trials,), np.zeros(n_trials,), np.zeros(n_trials,)
                for j in range(n_trials):
                    accuracies[j] += model_result[task_name][filter_key]['accuracy'][j]
                    perplexities[j] += model_result[task_name][filter_key]['perplexity'][j]
                    losses[j] += model_result[task_name][filter_key]['loss'][j]
                acc, acc_ci, acc_margin = mean_and_confidence_interval(accuracies)
                perp, perp_ci, perp_margin = mean_and_confidence_interval(perplexities)
                loss, loss_ci, loss_margin = mean_and_confidence_interval(losses)

                print(f'{model_name} in {task_name} tasks - Accuracy: {acc:.3f} ± {acc_margin:.2f}, Perplexity: {perp:.3f} ± {perp_margin:.2f}, Loss: {loss:.3f} ± {loss_margin:.2f}')
