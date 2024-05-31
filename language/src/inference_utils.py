import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F


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