import numpy as np
from typing import Optional, List
from tqdm import tqdm
from itertools import combinations

from joblib import Parallel, delayed

from helpers import randomized_argsort
from metric import Metric


def battle(sim):
    """Tries to distinguish the MEIs of two units from their similarities.
    Vectorized: sim can be either:
      - 2D (2K, 2K)          → single comparison, returns a scalar
      - 3D (P, 2K, 2K)       → batch of P comparisons, returns shape (P,)
    """
    batched = sim.ndim == 3
    if not batched:
        sim = sim[np.newaxis]          # (1, 2K, 2K)

    P, n, _ = sim.shape
    K = n // 2

    # ---- within-group mean similarities (exclude diagonal) ----------------
    eye = np.eye(K, dtype=bool)

    block_aa = sim[:, :K, :K]          # (P, K, K)
    block_bb = sim[:, K:, K:]
    block_ab = sim[:, :K, K:]
    block_ba = sim[:, K:, :K]

    # zero out diagonal then mean over K-1 neighbours
    block_aa = block_aa * (~eye)
    block_bb = block_bb * (~eye)

    a_sim_a = block_aa.sum(axis=2) / (K - 1)   # (P, K)
    b_sim_b = block_bb.sum(axis=2) / (K - 1)
    a_sim_b = block_ab.mean(axis=2)             # (P, K)
    b_sim_a = block_ba.mean(axis=2)

    acc_a = (a_sim_a > a_sim_b)                 # (P, K) bool
    acc_b = (b_sim_b > b_sim_a)
    acc = np.concatenate([acc_a, acc_b], axis=1).mean(axis=1)   # (P,)

    return acc if batched else float(acc[0])


# ---------------------------------------------------------------------------
# Helper: process one (i, j) pair — designed for joblib dispatch
# ---------------------------------------------------------------------------

def _process_pair(
    i: int,
    j: int,
    ind_top_per_quantile,      # list of length len(quantiles): each entry = (2K,) indices
    inputs: np.ndarray,
    activations_i: np.ndarray,
    activations_j: np.ndarray,
    metrics: dict,
    quantiles: List[float],
    output_keys: List[str],
):
    """Compute cross_mis for a single (i,j) unit pair."""
    result_pair = {k: np.zeros(len(quantiles)) for k in output_keys}

    for k_index, ind_top in enumerate(ind_top_per_quantile):
        for key, metric in metrics.items():
            if metric.precomputed:
                similarities = metric.precomputed_similarity(ind_top, ind_top)
            else:
                similarities = metric.compute_similarity(inputs[ind_top], inputs[ind_top])

            if metric.num_scores > 1:
                assert similarities.ndim == 3
                for s in range(metric.num_scores):
                    if 'lpips' in key and s == 0:
                        result_pair['accuracy_lpips'][k_index] = battle(similarities[:, :, s])
                    else:
                        result_pair['accuracy_%s_%s' % (key, s)][k_index] = battle(similarities[:, :, s])
            else:
                assert similarities.ndim == 2
                result_pair['accuracy_%s' % key][k_index] = battle(similarities)

    return i, j, result_pair


# ---------------------------------------------------------------------------

def cross_mis(
        inputs: np.ndarray,
        activations_a: np.ndarray,
        activations_b: np.ndarray,
        metrics: dict[str, Metric],
        quantiles: Optional[List[float]] = None,
    ):
    """
    Conducts a cross-MIS experiment: can you distinguish two units' MEIs?

    Parameters:
    inputs (np.ndarray): The input data for the experiment.
    activations_a (np.ndarray): The activations of unit A (1-D).
    activations_b (np.ndarray): The activations of unit B (1-D).
    metrics (dict[str, Metric]): The metrics to use.
    quantiles (List[float]): The % top MEIs to consider. Defaults to [0.01..0.05].

    Returns:
    dict: accuracy arrays keyed by metric name.
    """
    if quantiles is None:
        quantiles = [0.0025, 0.005, 0.01]

    if activations_a.ndim != 1:
        raise ValueError("activations_a must be 1-D, got shape %s" % list(activations_a.shape))
    if activations_b.ndim != 1:
        raise ValueError("activations_b must be 1-D, got shape %s" % list(activations_b.shape))
    if activations_a.shape != activations_b.shape:
        raise ValueError("activations_a and activations_b must have the same shape.")

    output = {}
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

    ind_top_a = randomized_argsort(-activations_a)
    ind_top_b = randomized_argsort(-activations_b)

    for k_index, K_percent in enumerate(quantiles):
        K = int(activations_a.shape[0] * K_percent)
        ind_top = np.concatenate([ind_top_a[:K], ind_top_b[:K]])

        for key, metric in metrics.items():
            if metric.precomputed:
                similarities = metric.precomputed_similarity(ind_top, ind_top)
            else:
                similarities = metric.compute_similarity(inputs[ind_top], inputs[ind_top])

            if metric.num_scores > 1:
                assert similarities.ndim == 3
                for i in range(metric.num_scores):
                    if 'lpips' in key and i == 0:
                        output['accuracy_lpips'][k_index] = battle(similarities[:, :, i])
                    else:
                        output['accuracy_%s_%s' % (key, i)][k_index] = battle(similarities[:, :, i])
            else:
                assert similarities.ndim == 2
                output['accuracy_%s' % key][k_index] = battle(similarities)

    return output


# ---------------------------------------------------------------------------

def compute_score(
        inputs: np.ndarray,
        activations: np.ndarray,
        metrics: dict[str, Metric],
        quantiles: Optional[List[float]] = None,
        n_jobs: int = -1,
        ):
    """
    Conducts a psychophysics experiment on all units.
    See https://arxiv.org/pdf/2307.05471.pdf Appendix A.2

    Parameters:
    inputs (np.ndarray): The input data for the experiment.
    activations (np.ndarray): (num_data, num_unit) activation matrix.
    metrics (dict[str, Metric]): The metrics to use.
    quantiles (List[float]): Top-% MEIs to consider. Defaults to [0.01..0.05].
    n_jobs (int): joblib parallel workers. -1 = all CPUs.

    Returns:
    dict: A dict containing accuracy arrays of shape (num_unit, num_unit, len(quantiles)).
    """
    assert activations.ndim == 2, "Activations must be 2-D, got shape %s." % list(activations.shape)
    num_data, num_unit = activations.shape

    if inputs.shape[0] != num_data:
        raise ValueError("inputs and activations must share the first dimension.")
    if quantiles is None:
        quantiles = [0.0025, 0.005, 0.01]
    if not all(q1 <= q2 for q1, q2 in zip(quantiles[:-1], quantiles[1:])):
        raise ValueError("quantiles must be in ascending order.")
    if int(quantiles[0] * num_data) < 2:
        raise ValueError("First quantile must yield >= 2 samples.")
    if quantiles[-1] > num_data // 2:
        raise ValueError("Last quantile must be < half the data (max %d)." % (num_data // 2))

    # ------------------------------------------------------------------ #
    # Pre-sort every unit's activations once  →  O(N log N) per unit      #
    # ------------------------------------------------------------------ #
    print("Pre-sorting activations for all units...")
    ind_top_all = [randomized_argsort(-activations[:, u]) for u in range(num_unit)]

    # For each quantile, precompute the per-unit top-K index arrays
    # Shape: quantiles × units → 1-D index array of length K
    K_values = [int(num_data * q) for q in quantiles]
    ind_top_per_unit_quantile = [
        [ind_top_all[u][:K] for u in range(num_unit)]
        for K in K_values
    ]

    # Build output key list (mirrors original logic)
    output_keys = []
    for key, metric in metrics.items():
        if metric.num_scores > 1:
            for i in range(metric.num_scores):
                if 'lpips' in key and i == 0:
                    output_keys.append('accuracy_lpips')
                else:
                    output_keys.append('accuracy_%s_%s' % (key, i))
        else:
            output_keys.append('accuracy_%s' % key)

    result = {k: np.zeros((num_unit, num_unit, len(quantiles))) for k in output_keys}
    result['quantiles'] = np.array(quantiles)

    # ------------------------------------------------------------------ #
    # Build the list of (i,j) pairs and their combined index arrays       #
    # ------------------------------------------------------------------ #
    pairs = list(combinations(range(num_unit), 2))   # N*(N-1)/2 pairs

    # For each pair (i,j) and each quantile, concatenate top-K indices
    def build_ind_top_per_quantile(i, j):
        return [
            np.concatenate([ind_top_per_unit_quantile[q_idx][i],
                            ind_top_per_unit_quantile[q_idx][j]])
            for q_idx in range(len(quantiles))
        ]

    # ------------------------------------------------------------------ #
    # Parallel dispatch across all pairs                                  #
    # ------------------------------------------------------------------ #
    print("Running %d unit-pair comparisons in parallel (n_jobs=%s)..." % (len(pairs), n_jobs))

    job_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_pair)(
            i, j,
            build_ind_top_per_quantile(i, j),
            inputs,
            activations[:, i],
            activations[:, j],
            metrics,
            quantiles,
            output_keys,
        )
        for i, j in tqdm(pairs, desc="pairs")
    )

    # ------------------------------------------------------------------ #
    # Collect results and fill symmetric entries in one pass              #
    # ------------------------------------------------------------------ #
    for i, j, pair_output in job_results:
        for key in output_keys:
            result[key][i, j] = pair_output[key]
            result[key][j, i] = pair_output[key]   # symmetry

    return result