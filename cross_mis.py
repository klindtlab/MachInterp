import numpy as np
from typing import Optional, List
from tqdm import tqdm

from helpers import randomized_argsort
from metric import Metric


def battle(sim):
    """Tries to distinguish the MEIs to two units from their similarities."""
    K = sim.shape[0] // 2
    a_sim_a = (sim[:K, :K].sum(1) - 1) / (K - 1)  # remove self comparison
    a_sim_b = sim[:K, K:].mean(1)
    b_sim_a = sim[K:, :K].mean()
    b_sim_b = (sim[K:, K:].sum(1) - 1) / (K - 1)  # remove self comparison
    logit_correct = a_sim_a + b_sim_b
    logit_wrong = a_sim_b + b_sim_a
    delta = logit_correct - logit_wrong
    return delta


def cross_mis(
        inputs: np.ndarray,
        activations_a: np.ndarray,
        activations_b: np.ndarray,
        metrics: dict[str, Metric],
        ks: Optional[List[int]] = [2, 4, 6, 8, 16],
    ):
    """
    Conducts a cross-MIS experiment: can you distinguish two units MEIs

    Parameters:
    inputs (np.ndarray): The input data for the experiment.
    activations_a (np.ndarray): The activations of a unit.
    activations_b (np.ndarray): The activations of a unit.
    metrics (dict[str, Metric]): The metrics to use.
    ks (List[int], optional): The k top MEIs to consider. Defaults to [2, 4, 6, 8, 16].

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

    ind_top_a = randomized_argsort(- activations_a)
    ind_top_b = randomized_argsort(- activations_b)
    for k_index, K in enumerate(ks):
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
                        output['accuracy_lpips'][k_index] = np.mean(
                            battle(similarities[:, :, i]) > 0)
                    else:
                        output['accuracy_%s_%s' % (key, i)][k_index] = np.mean(
                            battle(similarities[:, :, i]) > 0)
            else:
                assert len(similarities.shape) == 2
                output['accuracy_%s' % key][k_index] = np.mean(battle(similarities) > 0)
    return output


def compute_score(
        inputs: np.ndarray,
        activations: np.ndarray,
        metrics: dict[str, Metric],
        ks: Optional[List[int]] = None,
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
    if not all(q1 <= q2 for q1, q2 in zip(ks[:-1], ks[1:])):
        raise ValueError("Ks must be in ascending order.")
    if type(ks) == type(None):
        ks = 2 ** np.arange(1, int(np.ceil(np.log2(num_data // 2))))
    if ks[0] < 2:
        raise ValueError("First k must be >= 2.")
    if ks[-1] > num_data // 2:
        raise ValueError("Last k must be less than half the data = %s." % (num_data // 2))

    result = {}
    for m in metrics:
        result['accuracy_%s' % m] = np.zeros((num_unit, num_unit, len(ks)))
    for i in tqdm(range(num_unit)):
        for j in range(i + 1, num_unit):
            output = cross_mis(
                inputs=inputs, 
                activations_a=activations[:, i], 
                activations_b=activations[:, j], 
                metrics=metrics, 
                ks=ks
            )
            for key in output:
                result[key][i, j] = output[key]
    # fill all missing symmetrical comparisons
    for key in result:
        result[key] += np.transpose(result[key], (1, 0, 2))
    result['ks'] = np.array(ks)
    return result