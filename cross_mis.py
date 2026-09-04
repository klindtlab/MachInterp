import numpy as np
from typing import Optional, List
from tqdm import tqdm

from helpers import randomized_argsort
from metric import Metric


def battle(sim):
    """Tries to distinguish the MEIs to two units from their similarities.
    Compare every MEI from A to all other MEIs from A and all other MEIs
    from B and decide which ones are more similar (same, vice versa, for B).
    This gives two times the number of MEIs (K) comparisons that are right
    or wrong {0, 1}, the average over all is the final accuracy.
    """
    K = sim.shape[0] // 2
    a_sim_a = np.sum(sim[:K, :K] - np.diag(np.diag(sim[:K, :K])), axis=1) / (K - 1)  # remove self comparison
    b_sim_b = np.sum(sim[K:, K:] - np.diag(np.diag(sim[K:, K:])), axis=1) / (K - 1)  # remove self comparison
    a_sim_b = sim[:K, K:].mean(1)
    b_sim_a = sim[K:, :K].mean(1)
    acc_a = a_sim_a > a_sim_b
    acc_b = b_sim_b > b_sim_a
    acc = np.concatenate([acc_a, acc_b]).mean() # average over all
    return acc


def cross_mis(
        inputs: np.ndarray,
        activations_a: np.ndarray,
        activations_b: np.ndarray,
        metrics: dict[str, Metric],
        quantiles: Optional[List[float]] = [0.01, 0.02, 0.03, 0.04, 0.05],
    ):
    """
    Conducts a cross-MIS experiment: can you distinguish two units MEIs

    Parameters:
    inputs (np.ndarray): The input data for the experiment.
    activations_a (np.ndarray): The activations of a unit.
    activations_b (np.ndarray): The activations of a unit.
    metrics (dict[str, Metric]): The metrics to use.
    quantiles (List[float], optional): The % top MEIs to consider. Defaults to [0.01, 0.02, 0.03, 0.04, 0.05].

    Returns:
    The accuracy of the experiment.
    """
    if len(activations_a.shape) != 1:
        raise ValueError("Activations must be a vector, but have shape %s." % list(activations_a.shape))
    if len(activations_b.shape) != 1:
        raise ValueError("Activations must be a vector, but have shape %s." % list(activations_b.shape))
    if activations_a.shape[0] != activations_b.shape[0]:
        raise ValueError("Activations must have same shape.")
        
    output = dict()
    get_array = lambda: np.zeros(len(quantiles))
    for key, metric in metrics.items():
        if metric.num_scores > 1:
            for i in range(metric.num_scores):
                if 'lpips' in key and i == 0:
                    output['accuracy_lpips'] = get_array()
                else:
                    output['accuracy_%s_%s' % (key, i)] = get_array()
        else:
            output['accuracy_%s' % key] = get_array()

    ind_top_a = randomized_argsort(- activations_a)
    ind_top_b = randomized_argsort(- activations_b)
    for k_index, K_percent in enumerate(quantiles):
        #convert quantile to K val
        K = int(activations_a.shape[0]*K_percent)
        ind_top = np.concatenate([ind_top_a[:K], ind_top_b[:K]])
        # Calculate similarities for each metric
        for key, metric in metrics.items():
            if metric.precomputed:
                similarities = metric.precomputed_similarity(ind_top, ind_top)
            else:
                similarities = metric.compute_similarity(inputs[ind_top], inputs[ind_top])
            if metric.num_scores > 1:
                assert len(similarities.shape) == 3
                for i in range(metric.num_scores):
                    if 'lpips' in key and i == 0:
                        output['accuracy_lpips'][k_index] = battle(similarities[:, :, i])
                    else:
                        output['accuracy_%s_%s' % (key, i)][k_index] = battle(similarities[:, :, i])
            else:
                assert len(similarities.shape) == 2
                output['accuracy_%s' % key][k_index] = battle(similarities)
    return output


def compute_score(
        inputs: np.ndarray,
        activations: np.ndarray,
        metrics: dict[str, Metric],
        quantiles: Optional[List[float]] = None,
        ):
    """
    Conducts a psychophysics experiment on all units.
    See https://arxiv.org/pdf/2307.05471.pdf Appendix A.2

    Parameters:
    inputs (np.ndarray): The input data for the experiment.
    activations (np.ndarray): The activations of a unit.
    metrics (dict[str, Metric]): The metrics to use.
    ks (List[int], optional): The k top MEIs to consider. Defaults to all powers of 2 up to half the data.

    Returns:
    dict: A dict containing the logits and accuracy of the experiment.
    """
    assert len(activations.shape) == 2, "Activations must be a matrix, but have shape %s." % list(activations.shape)
    num_data, num_unit = activations.shape
    if inputs.shape[0] != num_data:
        raise ValueError("Input and activations must have the same first dimension.")
    if type(quantiles) == type(None):
        quantiles = [0.01, 0.02, 0.03, 0.04, 0.05]
    if not all(q1 <= q2 for q1, q2 in zip(quantiles[:-1], quantiles[1:])):
        raise ValueError("quantiles must be in ascending order.")
    if int(quantiles[0]*num_data) < 2:
        raise ValueError("First quantile must be >= 2.")
    if quantiles[-1] > num_data // 2:
        raise ValueError("Last quantile must be less than half the data = %s." % (num_data // 2))

    result = {}
    for m in metrics:
        result['accuracy_%s' % m] = np.zeros((num_unit, num_unit, len(quantiles)))
        if m == 'lpips':
            for i in range(1, 6):
                result['accuracy_%s_%s' % (m, i)] = np.zeros((num_unit, num_unit, len(quantiles)))
    for i in tqdm(range(num_unit)):
        for j in range(i + 1, num_unit):
            output = cross_mis(
                inputs=inputs, 
                activations_a=activations[:, i], 
                activations_b=activations[:, j], 
                metrics=metrics, 
                quantiles=quantiles
            )
            for key in output:
                result[key][i, j] = output[key]
    # fill all missing symmetrical comparisons
    for key in result:
        result[key] += np.transpose(result[key], (1, 0, 2))
    result['quantiles'] = np.array(quantiles)
    return result
