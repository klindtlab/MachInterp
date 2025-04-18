import numpy as np
from typing import Optional, List
from tqdm import tqdm

from helpers import randomized_argsort
from metric import Metric


def odd_one_out(
        inputs: np.ndarray,
        activations: np.ndarray,
        metrics: dict[str, Metric],
        ks: Optional[List[int]] = [2, 4, 6, 8, 16],
    ):
    """
    Conducts an odd one out experiment on a single unit.
    Chang, Blei 2003

    Parameters:
    inputs (np.ndarray): The input data for the experiment.
    activations (np.ndarray): The activations of a unit.
    metrics (dict[str, Metric]): The metrics to use.
    ks (List[int], optional): The k top MEIs to consider. Defaults to [2, 4, 6, 8, 16].

    Returns:
    The accuracy of the experiment.
    """
    if len(activations.shape) != 1:
        raise ValueError("Activations must be a vector, but have shape %s." % list(activations.shape))
        
    output = dict()
    get_array = lambda: np.zeros(len(ks))
    for key, metric in metrics.items():
        if metric.num_scores > 1:
            for i in range(metric.num_scores):
                if 'lpips' in key and i == 0:
                    output['accuracy_lpips'] = get_array()
                else:
                    output['accuracy_%s_%s' % (key, i)] = get_array()
        else:
            output['accuracy_%s' % key] = get_array()

    ind_top = randomized_argsort(- activations)
    for k_index, K in enumerate(ks):
        # Calculate similarities for each metric
        for key, metric in metrics.items():
            if metric.precomputed:
                similarities = metric.precomputed_similarity(ind_top, ind_top[:K])
            else:
                similarities = metric.compute_similarity(inputs[ind_top], inputs[ind_top[:K]])
            threshold = similarities[:K].mean()
            if metric.num_scores > 1:
                for i in range(metric.num_scores):
                    if 'lpips' in key and i == 0:
                        output['accuracy_lpips'][k_index] = np.mean(
                            similarities[K:, :, i].mean(1) < threshold)
                    else:
                        output['accuracy_%s_%s' % (key, i)][k_index] = np.mean(
                            similarities[K:, :, i].mean(1) < threshold)
            else:
                output['accuracy_%s' % key][k_index] = np.mean(
                    similarities[K:].mean(1) < threshold)
    return output


def compute_score(
        inputs: np.ndarray,
        activations: np.ndarray,
        metrics: dict[str, Metric],
        ks: Optional[List[int]] = [2, 4, 6, 8, 16],
        ):
    """
    Conducts an odd one out experiment on all single unit.
    Chang, Blei 2003

    Parameters:
    inputs (np.ndarray): The input data for the experiment.
    activations (np.ndarray): The activations of a unit.
    metrics (dict[str, Metric]): The metrics to use.
    ks (List[int], optional): The k top MEIs to consider. Defaults to [2, 4, 6, 8, 16].

    Returns:
    dict: A dict containing the accuracy of the experiment.
    """
    assert len(activations.shape) == 2, "Activations must be a matrix, but have shape %s." % list(activations.shape)
    num_data, num_unit = activations.shape
    if inputs.shape[0] != num_data:
        raise ValueError("Input and activations must have the same first dimension.")
    if not all(q1 <= q2 for q1, q2 in zip(ks[:-1], ks[1:])):
        raise ValueError("Ks must be in ascending order.")
    if ks[0] < 2:
        raise ValueError("First k must be >= 2.")
    if ks[-1] > num_data // 2:
        raise ValueError("Last k must be less than half the data = %s." % (num_data // 2))
    
    result = {'ks': np.array(ks)}
    for i in tqdm(range(num_unit)):
        output = odd_one_out(
            inputs=inputs, 
            activations=activations[:, i], 
            metrics=metrics, 
            ks=ks,
        )
        for key in output:
            if len(result) == 0:
                result[key] = []
            result[key].append(output[key])
            if i == num_unit - 1:
                result[key] = np.stack(result[key], 0)

    return result