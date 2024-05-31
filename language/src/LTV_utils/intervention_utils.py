from baukit import TraceDict


def add_learnable_task_vector(lt_vector, idx=-1, scale=1.0):
    """
    Adds a vector to the output of a specified layer in the model

    Parameters:
    edit_layer: the layer to perform the FV intervention
    fv_vector: the function vector to add as an intervention
    device: device of the model (cuda gpu or cpu)
    idx: the token index to add the function vector at

    Returns:
    add_act: a fuction specifying how to add a function vector to a layer's output hidden state
    """

    def add_act(output, layer_name):
        layer_idx = int(layer_name.split(".")[2])
        if isinstance(output, tuple):
            output[0][:, idx] += scale * lt_vector[layer_idx, :]
            return output
        else:
            return output

    return add_act


def ltv_intervention(sentence, target, lt_vector, model, model_config, tokenizer,
                     compute_nll=False,
                     generate_str=False):
    """
    Runs the model on the sentence and adds the function_vector to the output of edit_layer as a model intervention, predicting a single token.
    Returns the output of the model with and without intervention.

    Parameters:
    sentence: the sentence to be run through the model
    target: expected response of the model (str, or [str])
    edit_layer: layer at which to add the function vector
    function_vector: torch vector that triggers execution of a task
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    compute_nll: whether to compute the negative log likelihood of a teacher-forced completion (used to compute perplexity (PPL))
    generate_str: whether to generate a string of tokens or predict a single token

    Returns:
    fvi_output: a tuple containing output results of a clean run and intervened run of the model
    """
    # Clean Run, No Intervention:
    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    original_pred_idx = len(inputs.input_ids.squeeze()) - 1

    if compute_nll:
        target_completion = "".join(sentence + target[0])
        nll_inputs = tokenizer(target_completion, return_tensors='pt').to(device)
        nll_targets = nll_inputs.input_ids.clone()
        target_len = len(nll_targets.squeeze()) - len(inputs.input_ids.squeeze())
        nll_targets[:,
        :-target_len] = -100  # This is the accepted value to skip indices when computing loss (see nn.CrossEntropyLoss default)
        output = model(**nll_inputs, labels=nll_targets)
        clean_nll = output.loss.item()
        clean_output = output.logits[:, original_pred_idx, :]
    elif generate_str:
        MAX_NEW_TOKENS = 16
        output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1, max_new_tokens=MAX_NEW_TOKENS)
        clean_output = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])
    else:
        clean_output = model(**inputs).logits[:, -1, :]

    # Perform Intervention
    # Repeat the learnable task vector
    intervention_fn = add_learnable_task_vector(lt_vector, scale=1.0)
    with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn):
        if compute_nll:
            output = model(**nll_inputs, labels=nll_targets)
            intervention_nll = output.loss.item()
            intervention_output = output.logits[:, original_pred_idx, :]
        elif generate_str:
            output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1,
                                    max_new_tokens=MAX_NEW_TOKENS)
            intervention_output = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])
        else:
            intervention_output = model(**inputs).logits[:, -1, :]  # batch_size x n_tokens x vocab_size, only want last token prediction

    fvi_output = (clean_output, intervention_output)
    if compute_nll:
        fvi_output += (clean_nll, intervention_nll)

    return fvi_output


def ltv_intervention_natural_text(sentence, lt_vector, model, model_config, tokenizer,
                                 max_new_tokens=16, num_interv_tokens=None, do_sample=False):
    """
    Allows for intervention in natural text where we generate and intervene on several tokens in a row.

    Parameters:
    sentence: sentence to intervene on with the FV
    edit_layer: layer at which to add the function vector
    function_vector: vector to add to the model that triggers execution of a task
    model: huggingface model
    model_config: dict with model config parameters (n_layers, n_heads, etc.)
    tokenizer: huggingface tokenizer
    max_new_tokens: number of tokens to generate
    num_interv_tokens: number of tokens to apply the intervention for (defaults to all subsequent generations)
    do_sample: whether to sample from top p tokens (True) or have deterministic greedy decoding (False)

    Returns:
    clean_output: tokens of clean output
    intervention_output: tokens of intervention output

    """
    # Clean Run, No Intervention:
    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    clean_output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)

    # Perform Intervention
    intervention_fn = add_learnable_task_vector(lt_vector)

    if num_interv_tokens is not None and num_interv_tokens < max_new_tokens:  # Intervene only for a certain number of tokens
        num_extra_tokens = max_new_tokens - num_interv_tokens
        with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn):
            intervention_output = model.generate(**inputs, max_new_tokens=num_interv_tokens, do_sample=do_sample,
                                                 pad_token_id=tokenizer.eos_token_id)
        intervention_output = model.generate(intervention_output, max_new_tokens=num_extra_tokens,
                                             pad_token_id=tokenizer.eos_token_id, do_sample=do_sample)
    else:
        with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn):
            intervention_output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                                 pad_token_id=tokenizer.eos_token_id)

    return clean_output, intervention_output
