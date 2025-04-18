import numpy as np
from typing import Optional, List
from tqdm import tqdm

from helpers import randomized_argsort, get_extreme_ind
from metric import Metric


def extract_logits(similarities, zscore, pool_fun):
    num_references = similarities.shape[0] // 2
    if zscore:
        similarities -= np.mean(similarities)
        std = np.std(similarities)
        if np.isnan(std) or std == 0.0:
            std = 1.0
        similarities /= std
    logits = np.zeros((2, 2))
    # compute logit for top
    logits[0, 0] = pool_fun(similarities[:num_references, 0])  # evidence_from_top
    logits[0, 1] = pool_fun(similarities[num_references:, 1])  # evidence_from_bottom
    # compute logit bottom
    logits[1, 0] = pool_fun(similarities[:num_references, 1])  # evidence_from_top
    logits[1, 1] = pool_fun(similarities[num_references:, 0])  # evidence_from_bottom
    return logits


def compute_accuracy(logits):
    actual_logits = logits.sum(-1)  # summing evidences: Q x T x 2
    accuracy = np.mean(actual_logits[:, :, 0] - actual_logits[:, :, 1] > 0, axis=1)
    # score = np.mean(softmax(actual_logits)[:, :, 0], axis=1)  # probablity of correct answer: Q
    return accuracy


def run_single_unit_psychophysics(
        inputs: np.ndarray,
        activations: np.ndarray,
        metrics: dict[str, Metric],
        quantiles: Optional[List[float]] = [0.0, 0.25],
        num_trials: int = 20, 
        num_references: int = 9,
        pool_fun: callable = np.mean,  # min/max would not make sense with label
        zscore: bool = True,
        seed: int = 42):
    """
    Conducts a psychophysics experiment on a single unit.
    See https://arxiv.org/pdf/2307.05471.pdf Appendix A.2

    Parameters:
    inputs (np.ndarray): The input data for the experiment.
    activations (np.ndarray): The activations of a unit.
    metrics (dict[str, Metric]): The metrics to use.
    quantiles (List[float], optional): The quantiles to consider. Defaults to [0.0, 0.25].
    num_trials (int, optional): The number of trials to conduct. Defaults to 20.
    num_references (int, optional): The number of reference units to consider. Defaults to 9.
    pool_function (callable, optional): The function to use for pooling. Defaults to np.mean.
    zscore (bool, optional): Whether to z-score (subtract mean and divide by std.dev.) similarity measures.
    seed (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
    dict: A tuple containing the logits and accuracy of the experiment.
    """
    if len(activations.shape) != 1:
        raise ValueError("Activations must be a vector, but have shape %s." % list(activations.shape))
        
    np.random.seed(seed)
    output = dict()
    for key, metric in metrics.items():
        if metric.num_scores > 1:
            for i in range(metric.num_scores):
                if 'lpips' in key and i == 0:
                    output['logits_lpips'] = np.zeros((len(quantiles), num_trials, 2, 2))  # Q x T x 2 x 2
                else:
                    output['logits_%s_%s' % (key, i)] = np.zeros((len(quantiles), num_trials, 2, 2))  # Q x T x 2 x 2
        else:
            output['logits_%s' % key] = np.zeros((len(quantiles), num_trials, 2, 2))  # Q x T x 2 x 2

    for quantile_index, quantile in enumerate(quantiles):
        ind = get_extreme_ind(activations, quantile)
        y = activations.copy()
        y[ind] = np.median(y)
        
        # Get sorted indices
        ind_sort_bottom = randomized_argsort(y)
        ind_sort_top = ind_sort_bottom[::-1]
        
        # Get query indices by shuffling the top num_trials
        ind_query_top = np.random.permutation(ind_sort_top[:num_trials])
        ind_query_bottom = np.random.permutation(ind_sort_bottom[:num_trials])

        # Get reference indices
        ind_reference_start = num_trials + np.arange(num_references) * num_trials

        for trial_index in range(num_trials):
            # Get reference indices for this trial
            ind_reference_top = ind_sort_top[ind_reference_start + trial_index]
            ind_reference_bottom = ind_sort_bottom[ind_reference_start + trial_index]
            
            # Combine indices for batch processing
            ind_reference = np.concatenate([ind_reference_top, ind_reference_bottom])
            ind_query = np.array([ind_query_top[trial_index], ind_query_bottom[trial_index]])
            
            # Calculate similarities for each metric
            for key, metric in metrics.items():
                if metric.precomputed:
                    similarities = metric.precomputed_similarity(ind_reference, ind_query)
                else:
                    similarities = metric.compute_similarity(inputs[ind_reference], inputs[ind_query])
                if metric.num_scores > 1:
                    for i in range(metric.num_scores):
                        if 'lpips' in key and i == 0:
                            output['logits_lpips'][quantile_index, trial_index] = extract_logits(
                                similarities[:, :, i], zscore, pool_fun)
                        else:
                            output['logits_%s_%s' % (key, i)][quantile_index, trial_index] = extract_logits(
                                similarities[:, :, i], zscore, pool_fun)
                else:
                    output['logits_%s' % key][quantile_index, trial_index] = extract_logits(
                        similarities, zscore, pool_fun)
                
    # compute accuracy
    keys = list(output.keys())
    for key in keys:
        if key.startswith('logits'):
            output['accuracy' + key[6:]] = compute_accuracy(output[key])
        
    return output


def compute_score(
        inputs: np.ndarray,
        activations: np.ndarray,
        metrics: dict[str, Metric],
        quantiles: Optional[List[float]] = [0, .1, .2, .3, .4, .5], 
        num_trials: int = 50, 
        num_references: int = 9,
        pool_fun: callable = np.mean,  # min/max would not make sense with label
        zscore: bool = True,
        seed: int = 42
        ):
    """
    Conducts a psychophysics experiment on all units.
    See https://arxiv.org/pdf/2307.05471.pdf Appendix A.2

    Parameters:
    inputs (np.ndarray): The input data for the experiment.
    activations (np.ndarray): The activations of a unit.
    metrics (dict[str, Metric]): The metrics to use.
    quantiles (List[float], optional): The quantiles to consider. Defaults to [0.0, 0.25].
    num_trials (int, optional): The number of trials to conduct. Defaults to 20.
    num_references (int, optional): The number of reference units to consider. Defaults to 9.
    pool_function (callable, optional): The function to use for pooling. Defaults to np.mean.
    zscore (bool, optional): Whether to z-score (subtract mean and divide by std.dev.) similarity measures.
    seed (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
    dict: A dict containing the logits and accuracy of the experiment.
    """
    assert len(activations.shape) == 2, "Activations must be a matrix, but have shape %s." % list(activations.shape)
    num_data, num_unit = activations.shape
    if inputs.shape[0] != num_data:
        raise ValueError("Input and activations must have the same first dimension.")
    if not all(q1 <= q2 for q1, q2 in zip(quantiles, quantiles[1:])):
        raise ValueError("Quantiles must be in ascending order.")
    if quantiles[0] < 0.0:
        raise ValueError("First quantile must be >= 0.0.")
    if quantiles[-1] > 0.5:
        raise ValueError("Last quantile must be <= 0.5.")
    max_trials = num_data // ((num_references + 1) * 2)
    if num_data <= num_trials * (num_references + 1) * 2:
        raise ValueError("Not enough data for the specified number of trials and references. At most num_trials=%s." % max_trials)
    
    result = {'quantiles': quantiles}
    for i in tqdm(range(num_unit)):
        output = run_single_unit_psychophysics(
            inputs=inputs, 
            activations=activations[:, i], 
            metrics=metrics, 
            quantiles=quantiles,
            num_trials=num_trials,
            num_references=num_references,
            pool_fun=pool_fun,
            zscore=zscore,
            seed=seed,
        )
        for key in output:
            if i == 0:
                result[key] = []
            result[key].append(output[key])
            if i == num_unit - 1:
                result[key] = np.stack(result[key], 0)
    return result