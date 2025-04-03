import torch
from torch import Tensor
from PIL import Image
# from metric import get_metric
import math
import numpy as np


def randomized_argsort(arr, descending=False):
    """
    Sort array with random tie-breaking.
    
    Args:
        arr: Array to sort
        descending: Whether to sort in descending order
        
    Returns:
        Array of indices that would sort the input array
    """
    arr = np.asarray(arr)
    # Create a random key of the same length
    rand_key = np.random.rand(arr.shape[0])
    # np.lexsort uses the last key as the primary sort
    # Here, the primary key is the array itself and the secondary key is random
    idx = np.lexsort((rand_key, arr))
    if descending:
        idx = idx[::-1]
    return idx


def sample_random_indices(n_units, N, K, subset_length):
    """
    Generate random indices for sampling without replacement.
    
    Args:
        n_units: Number of units
        N: Number of tasks
        K: Number of samples to draw
        subset_length: Length of the subset to sample from
        
    Returns:
        Array of shape (n_units, N, K) containing randomly sampled indices
    """
    # Use broadcasting to generate indices for all units and tasks at once
    indices = np.random.choice(subset_length, size=(n_units, N, K), replace=False)
    return indices

def get_subset(array, indices):
    """
    Get subset of array using indices.
    
    Args:
        array: Array to index into
        indices: Array of indices
        
    Returns:
        Subset of array indexed by indices
    """
    # Use advanced indexing to get subsets efficiently
    return array[indices]

def get_activations_subset(activations, indices):
    """
    Get subset of activations using indices.
    
    Args:
        activations: Array of shape (n_units, n_samples)
        indices: Array of shape (n_units, N, K)
        
    Returns:
        Array of shape (n_units, N, K) containing selected activations
    """
    n_units, N, K = indices.shape
    result = np.empty((n_units, N, K))
    
    for i in range(n_units):
        for j in range(N):
            result[i, j] = activations[i, indices[i, j]]
    
    return result

def get_images_subset(images, indices):
    """
    Get subset of images using indices.
    
    Args:
        images: Array of shape (n_samples, *image_dims)
        indices: Array of shape (n_units, N, K)
        
    Returns:
        Array of shape (n_units, N, K, *image_dims) containing selected images
    """
    n_units, N, K = indices.shape
    image_dims = images.shape[1:]
    result = np.empty((n_units, N, K) + image_dims)
    
    for i in range(n_units):
        for j in range(N):
            result[i, j] = images[indices[i, j]]
    
    return result

def sample_activation_subsets(seed: int, activations, K: int, N: int, 
                            quantile: float | int, activations_sort_id=None):
    """
    Randomly generates indices of query and explanation images from appropriate quantile range
    for ensemble of tasks. 

    Args:
        seed: Seed for random generation
        activations: Array of shape (n_units x ds_length) containing activations
        K: Number of images in each explanation
        N: Number of tasks
        quantile: The quantile range to draw from
        activations_sort_id: Optional. Argsort of the activations for all units

    Returns:
        Tuple of (top_id, bottom_id):
        - top_id: Array of shape (n_units x N x (K+1)) containing indices of sampled positive query/explanation images
        - bottom_id: Array of shape (n_units x N x (K+1)) containing indices of sampled negative query/explanation images
    """
    np.random.seed(seed)
    
    n_units = activations.shape[0]
    n_samples = activations.shape[1]
    subset_length = math.ceil(n_samples * quantile)
    assert not subset_length < K+1

    # Generate random indices for top and bottom subsets
    top_id = sample_random_indices(n_units, N, K+1, subset_length)
    bottom_id = sample_random_indices(n_units, N, K+1, subset_length)

    if quantile == 1:
        return top_id, bottom_id

    if activations_sort_id is None:
        activations_sort_id = np.argsort(activations, axis=-1)

    # Get the top and bottom subsets from sorted indices
    top_set_id = np.flip(activations_sort_id, axis=-1)[:, :subset_length]
    bottom_set_id = activations_sort_id[:, :subset_length]
    
    # Map the random indices to actual indices in the sorted arrays
    top_id = get_subset(top_set_id, top_id)
    bottom_id = get_subset(bottom_set_id, bottom_id)

    return top_id, bottom_id

def sort_by_activation(top_id, bottom_id, activations):
    """
    Sort subsets of indices based on their activation values.

    Args:
        top_id: Array of indices for top activations (N_units x N x K+1)
        bottom_id: Array of indices for bottom activations (N_units x N x K+1) 
        activations: Array of activation values (N_units x N_images)

    Returns:
        Tuple of (top_id_sorted, bottom_id_sorted):
        - top_id_sorted: Indices sorted by decreasing activation values
        - bottom_id_sorted: Indices sorted by increasing activation values
    """
    # Get activation values for the indices
    top_act = get_subset(activations, top_id)
    bottom_act = get_subset(activations, bottom_id)
    
    # Sort indices based on activation values
    top_sort_id = np.argsort(top_act, axis=-1)[:, :, ::-1]  # descending order
    bottom_sort_id = np.argsort(bottom_act, axis=-1)  # ascending order
    
    # Apply sorting to the original indices
    top_id_sorted = get_subset(top_id, top_sort_id)
    bottom_id_sorted = get_subset(bottom_id, bottom_sort_id)
    
    return top_id_sorted, bottom_id_sorted

def generate_query_explanation_sets(seed: int, images, activations, 
                                  K: int, N: int, 
                                  quantile: float,
                                  activations_sort_id=None):
    """
    Generate query and explanations for ALL psychophysics tasks.

    Args:
        seed: Random seed for reproducibility
        images: Array of preprocessed images
        activations: Array of shape (N_units x N_images) containing activations
        K: Number of images in each explanation set per task
        N: Number of tasks per unit
        quantile: Quantile range over the activations for psychophysics query/explanation generation
        activations_sort_id: Optional. Argsort of the activations for all units

    Returns:
        Tuple of (query_set, explanation_set):
        - query_set: Tuple of (query_plus_set, query_minus_set) containing batched queries
        - explanation_set: Tuple of (explanation_plus_set, explanation_minus_set) containing batched explanations
    """
    # Get top and bottom indices
    top_id, bottom_id = sample_activation_subsets(seed, activations, K=K, 
                                                N=N, quantile=quantile,
                                                activations_sort_id=activations_sort_id)
    
    # Sort the indices based on activation values
    top_id, bottom_id = sort_by_activation(top_id, bottom_id, activations)

    # Get explanation sets (first K indices) and query sets (last index)
    explanation_plus_set = get_subset(images, top_id[:, :, :K])
    query_plus_set = get_subset(images, top_id[:, :, K:K+1]).squeeze(axis=2)

    explanation_minus_set = get_subset(images, bottom_id[:, :, :K])
    query_minus_set = get_subset(images, bottom_id[:, :, K:K+1]).squeeze(axis=2)

    query_set = (query_plus_set, query_minus_set)
    explanation_set = (explanation_plus_set, explanation_minus_set)

    return query_set, explanation_set

def calculate_similarity_scores(query_batch, explanation_batch, similarity_metric):
    """
    Calculate similarity between query and explanation images.

    Args:
        query_batch: Array of shape (N, *I_dim) containing preprocessed query images
        explanation_batch: Array of shape (N, K, *I_dim) containing preprocessed explanation images
        similarity_metric: Callable similarity metric function

    Returns:
        Array of shape (N,) containing aggregated similarity scores
    """
    # Calculate similarity for each query-explanation pair
    similarity_scores = np.array([similarity_metric(q, e) for q, e in zip(query_batch, explanation_batch)])
    # Aggregate similarities
    return np.mean(similarity_scores, axis=-1)

def calculate_unit_mis(query, explanation, similarity_metric, alpha=None):
    """
    Calculate Machine Interpretability Score (MIS) for SINGLE UNIT

    Args:
        query: Tuple of (q_plus, q_minus) containing queries for SINGLE UNIT
        explanation: Tuple of (e_plus, e_minus) containing explanations for SINGLE UNIT
        similarity_metric: Callable similarity metric function
        alpha: Parameter for Sigmoid function in MIS calculation. If None, defaults to unnormalized psychophysics accuracy.

    Returns:
        Scalar MIS of SINGLE UNIT
    """
    e_plus, e_minus = explanation
    q_plus, q_minus = query

    assert q_plus.shape[0] == e_plus.shape[0]
    assert q_minus.shape[0] == e_minus.shape[0]
    assert q_plus.shape[0] == q_minus.shape[0]

    # Calculate similarities
    s_plus_plus = calculate_similarity_scores(q_plus, e_plus, similarity_metric)
    s_plus_minus = calculate_similarity_scores(q_plus, e_minus, similarity_metric)
    s_minus_plus = calculate_similarity_scores(q_minus, e_plus, similarity_metric)
    s_minus_minus = calculate_similarity_scores(q_minus, e_minus, similarity_metric)

    # Calculate deltas
    delta_plus = s_plus_plus - s_plus_minus
    delta_minus = s_minus_plus - s_minus_minus
    delta_difference = delta_plus - delta_minus

    if alpha is None:
        # Unnormalized psychophysics accuracy
        return np.sum(delta_difference > 0) / len(delta_difference)

    # Sigmoid-based MIS
    mis = 1 / (1 + np.exp(-alpha * delta_difference))
    return np.mean(mis)

def calculate_all_units_mis(query_set, explanation_set, similarity_metric, alpha=0.16):
    """
    Calculate MIS for ALL UNITS

    Args:
        query_set: Tuple of (q_plus_set, q_minus_set) containing queries for ALL UNITS
        explanation_set: Tuple of (e_plus_set, e_minus_set) containing explanations for ALL UNITS
        similarity_metric: Callable similarity metric function
        alpha: Parameter for Sigmoid function in MIS calculation

    Returns:
        Array of shape (N_units,) containing MIS of ALL UNITS
    """
    assert len(query_set) == 2
    assert len(explanation_set) == 2
    assert query_set[0].shape[0] == explanation_set[0].shape[0]
    assert query_set[1].shape[0] == explanation_set[1].shape[0]
    assert query_set[0].shape[0] == query_set[1].shape[0]

    query_plus_set, query_minus_set = query_set
    explanation_plus_set, explanation_minus_set = explanation_set

    n_units = query_plus_set.shape[0]
    mis_scores = np.empty(n_units)

    # Calculate MIS for each unit
    for i in range(n_units):
        query = (query_plus_set[i], query_minus_set[i])
        explanation = (explanation_plus_set[i], explanation_minus_set[i])
        mis_scores[i] = calculate_unit_mis(query, explanation, similarity_metric, alpha)

    return mis_scores

def run_psychophysics_experiment(seed: int, task_data, metric_type: str,
                               K: int, N: int, quantile: float, alpha: float=None, metric=None):
    """
    Run psychophysics experiment to generate Machine Interpretability Score (MIS).
    
    Args:
        seed: Random seed for reproducibility
        task_data: Task configuration object containing image data and activations
        metric_type: Type of similarity metric to use
        K: Number of images in each explanation set per task
        N: Number of psychophysics experiment trials
        quantile: Quantile threshold for selecting images
        alpha: Optional threshold parameter for MIS calculation
        metric: Optional pre-initialized metric function
    
    Returns:
        Array of shape (N_units,) containing MIS of ALL UNITS
    """
    if metric is None:
        metric = get_metric(metric_type)

    # Generate query and explanation sets
    query_set, explanation_set = generate_query_explanation_sets(
        seed=seed,
        images=task_data.get_processed(metric_type=metric_type),
        activations=task_data.y_data,
        K=K,
        N=N,
        quantile=quantile,
        activations_sort_id=task_data.y_sort_id
    )
    
    # Calculate MIS for all units
    return calculate_all_units_mis(query_set, explanation_set, metric, alpha=alpha)


def get_center_ind(activations, quantile):
    """Get indices of all activations in center quantile, only use distribution center between quantiles."""
    return np.logical_and(
        activations >= np.quantile(activations, quantile), 
        activations <= np.quantile(activations, 1 - quantile))


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def run_psychophysics(
        inputs: np.ndarray,
        activations: np.ndarray,
        labels: Optional[np.ndarray] = None,
        quantiles: Optional[List[float]] = [0.0, 0.25]):
    """
    Conducts a psychophysics experiment on all units.
    See https://arxiv.org/pdf/2307.05471.pdf Appendix A.2

    Parameters:
    inputs (np.ndarray): The input data for the experiment.
    activations (np.ndarray): The activations of the units.
    metric_name (str): The name of the metric to use.
    metric (callable, optional): The metric function to use. If None, a metric function is obtained using get_metric.

    Returns:
    dict: A dict containing the logits and score of the experiment.
    """
    result = {}
    num_unit = activations.shape[1]

    print('Computing all image metrics')
    metric = get_metric('image')
    for i in tqdm(range(num_unit)):
        output = run_single_unit_psychophysics(
            inputs=inputs, activations=activations[:, i], 
            metric_name='image', metric=metric, quantiles=quantiles)
        for key in output:
            if i == 0:
                result[key] = []
            result[key].append(output[key])
            if i == num_unit - 1:
                result[key] = np.stack(result[key], 0)

    if type(labels) != type(None):
        if len(labels.shape) == 1:
            labels = labels[:, None]
        metric = get_metric('label')
        for label in range(labels.shape[1]):
            print('Computing metric for label', label)
            for i in tqdm(range(num_unit)):
                output = run_single_unit_psychophysics(
                    inputs=labels[:, label], activations=activations[:, i], 
                    metric_name='label', metric=metric, quantiles=quantiles)
                for key in output:
                    if i == 0:
                        result['label_%s_' % label + key] = []
                    result['label_%s_' % label + key].append(output[key])
                    if i == num_unit - 1:
                        result['label_%s_' % label + key] = np.stack(result['label_%s_' % label + key], 0)
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
        metric_name: str,
        metric: Optional[callable] = None,
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
    metric_name (str): The name of the metric to use.
    metric (callable, optional): The metric function to use. If None, a metric function is obtained using get_metric.
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
    if len(inputs.shape) == 4:  # images
        if not metric_name in ['l2', 'color', 'lpips', 'image', 'dreamsim']:
            raise ValueError("Invalid metric for image data.")
    elif len(inputs.shape) == 1:  # labels
        if metric_name != 'label':
            raise ValueError("Invalid metric for label data.")
    else:
        raise ValueError("Inputs of shape (%s) unclear (NCHW for images, NC for labels)." % list(inputs.shape))
    if metric is None:
        metric = get_metric(metric_name)
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
    if metric_name == 'image':
        random_image = np.random.normal(0, 1, (2, 3, 224, 224))
        tmp = metric(random_image, random_image)
        for key in tmp:
            output['logits_%s' % key] = np.zeros((len(quantiles), num_trials, 2, 2))  # Q x T x 2 x 2
    else:
        output['logits'] = np.zeros((len(quantiles), num_trials, 2, 2))  # Q x T x 2 x 2

    for quantile_index, quantile in enumerate(quantiles):
        ind = get_center_ind(activations, quantile)
        y = activations[ind].copy()
        x = inputs[ind].copy()
        
        ind_sort_bottom = np.argsort(y)
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
            if metric_name == 'image':
                for key in similarities:
                    output['logits_%s' % key][quantile_index, trial_index] = extract_logits(
                        similarities[key], zscore, pool_fun)
            else:
                output['logits'][quantile_index, trial_index] = extract_logits(
                    similarities, zscore, pool_fun)

    # compute score
    keys = list(output.keys())
    for key in keys:
        if key.startswith('logits'):
            output['score' + key[6:]] = compute_score(output[key])
        
    return output