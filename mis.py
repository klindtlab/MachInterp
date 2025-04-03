import numpy as np
from typing import Optional, List

import os
import sys
from tqdm import tqdm

from helpers import randomized_argsort, softmax
from metric import Metric

def get_center_ind(activations, quantile):
    """Get indices of all activations in center quantile, only use distribution center between quantiles."""
    return np.logical_and(
        activations >= np.quantile(activations, quantile), 
        activations <= np.quantile(activations, 1 - quantile))


def run_psychophysics(
        inputs: np.ndarray,
        activations: np.ndarray,
        quantiles: Optional[List[float]] = [0.0, 0.25],
        metric: Optional[Metric] = None,
        ):
    """
    Conducts a psychophysics experiment on all units.
    See https://arxiv.org/pdf/2307.05471.pdf Appendix A.2

    Parameters:
    inputs (np.ndarray): The input data for the experiment.
    activations (np.ndarray): The activations of the units.
    metric (callable, optional): The metric function to use. If None, a metric function is obtained using get_metric.

    Returns:
    dict: A dict containing the logits and score of the experiment.
    """
    result = {}
    num_unit = activations.shape[1]

    for i in tqdm(range(num_unit)):
        output = run_single_unit_psychophysics(
            inputs=inputs, activations=activations[:, i], 
            metric=metric, quantiles=quantiles)
        for key in output:
            if i == 0:
                result[key] = []
            result[key].append(output[key])
            if i == num_unit - 1:
                result[key] = np.stack(result[key], 0)
    return result


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


def compute_score(logits):
    actual_logits = logits.sum(-1)  # summing evidences: Q x T x 2
    score = np.mean(softmax(actual_logits)[:, :, 0], axis=1)  # probablity of correct answer: Q
    return score * 2 - 1


def run_single_unit_psychophysics(
        inputs: np.ndarray,
        activations: np.ndarray,
        metric: Optional[Metric] = None,
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
    activations (np.ndarray): The activations of the units.
    metric (Metric, optional): The metric function to use. If None, a metric function is obtained using get_metric.
    quantiles (List[float], optional): The quantiles to consider. Defaults to [0.0, 0.25].
    num_trials (int, optional): The number of trials to conduct. Defaults to 20.
    num_references (int, optional): The number of reference units to consider. Defaults to 9.
    pool_function (callable, optional): The function to use for pooling. Defaults to np.mean.
    zscore (bool, optional): Whether to z-score (subtract mean and divide by std.dev.) similarity measures.
    seed (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
    dict: A tuple containing the logits and score of the experiment.
    """
    if len(activations.shape) != 1:
        raise ValueError("Activations must be a vector, but have shape %s." % list(activations.shape))
    if inputs.shape[0] != activations.shape[0]:
        raise ValueError("Input and activations must have the same first dimension.")
    if not all(q1 <= q2 for q1, q2 in zip(quantiles, quantiles[1:])):
        raise ValueError("Quantiles must be in ascending order.")
    if quantiles[0] < 0.0:
        raise ValueError("First quantile must be >= 0.0.")
    if quantiles[-1] >= 0.5:
        raise ValueError("Last quantile must be < 0.5.")
    # make sure there is at least twice as many left as we need for highest quantile.
    if np.sum(get_center_ind(activations, quantiles[-1])) <= num_trials * (num_references + 1) * 2 * 2:
        raise ValueError("Not enough data for the specified number of quantiles, trials and references.")
        
    np.random.seed(seed)
    output = dict()
    output['logits'] = np.zeros((len(quantiles), num_trials, 2, 2))  # Q x T x 2 x 2

    for quantile_index, quantile in enumerate(quantiles):
        ind = get_center_ind(activations, quantile)
        y = activations[ind].copy()
        x = inputs[ind].copy()
        
        ind_sort_bottom = randomized_argsort(y)
        ind_sort_top = ind_sort_bottom[::-1]
        ind_query_top = np.random.choice(num_trials, num_trials, replace=False)
        ind_query_bottom = np.random.choice(num_trials, num_trials, replace=False)
        query_top = x[ind_sort_top[:num_trials][ind_query_top]]
        query_bottom = x[ind_sort_bottom[:num_trials][ind_query_bottom]]

        # permute only query, enough randomness, decreasing difficulty not a problem here
        ind_reference = num_trials + np.arange(num_references) * num_trials

        for trial_index in range(num_trials):
            reference_top = x[ind_sort_top[ind_reference + trial_index]]
            reference_bottom = x[ind_sort_bottom[ind_reference + trial_index]]
            
            # Run all together (twice as fast):
            reference = np.concatenate([reference_top, reference_bottom], 0)
            query = np.concatenate([query_top[trial_index:trial_index + 1], 
                                    query_bottom[trial_index:trial_index + 1]])
            similarities = metric(reference, query)
            output['logits'][quantile_index, trial_index] = extract_logits(
                similarities, zscore, pool_fun)

    # compute score
    keys = list(output.keys())
    for key in keys:
        if key.startswith('logits'):
            output['score' + key[6:]] = compute_score(output[key])
        
    return output