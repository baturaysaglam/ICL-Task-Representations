import random

import numpy as np
import torch

from src.utils.extract_utils import get_mean_head_activations
from src.ltv_utils.extract_utils import get_attn_out
from src.ltv_utils.intervention_utils import ltv_intervention
from src.utils.prompt_utils import word_pairs_to_prompt_data, create_prompt


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_attn_out(dataset, model, model_config, tokenizer, batch_size):
    activations = []
    for dataset_ in dataset:
        mean_activations, _ = get_mean_head_activations(dataset_, model, model_config, tokenizer, N_TRIALS=batch_size // len(dataset), batch_structure=True)
        activations.append(mean_activations)
    activations = torch.cat(activations, dim=0)
    mean_activations = activations.mean(dim=0)
    attn_out = get_attn_out(mean_activations.unsqueeze(0), model, model_config)
    return attn_out


def sample_data(datasets, n_examples, batch_size, shuffle_labels=True):
    sentences = [None] * batch_size
    targets = [None] * batch_size
    for i, dataset_ in enumerate(datasets):
        for sample_idx in range(batch_size // len(datasets)):
            test_idx = np.random.randint(0, len(dataset_['test']))

            word_pairs = dataset_['train'][np.random.choice(len(dataset_['train']), n_examples, replace=False)]
            test_pair = dataset_['test'][test_idx]

            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=test_pair, prepend_bos_token=True, shuffle_labels=shuffle_labels)
            sentence = create_prompt(prompt_data)
            sentences[sample_idx + i * batch_size // len(datasets)] = sentence
            targets[sample_idx + i * batch_size // len(datasets)] = test_pair['output']
    if len(datasets) > 1:
        indices = list(range(len(sentences)))
        random.shuffle(indices)
        sentences = [sentences[i] for i in indices]
        targets = [targets[i] for i in indices]
    return sentences, targets


def forward_pass(model, model_config, tokenizer, vocab_size, sentences, targets, lt_vector):
    batch_size = len(sentences)
    intervention_logits = torch.zeros(batch_size, vocab_size).to(model.device)
    clean_logits = torch.zeros_like(intervention_logits).to(model.device)
    target_indices = torch.zeros(batch_size).to(model.device)

    lt_vector = lt_vector.squeeze(0)
    for i, (sentence, target) in enumerate(zip(sentences, targets)):
        clean_output, intervention_output = ltv_intervention(sentence, target, lt_vector, model, model_config,
                                                             tokenizer, compute_nll=False, generate_str=False)
        intervention_logits[i] = intervention_output
        clean_logits[i] = clean_output
        target_indices[i] = tokenizer.encode(" " + target)[0]

    return intervention_logits, clean_logits, target_indices
