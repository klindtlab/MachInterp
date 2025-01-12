import torch
from torch import Tensor
from PIL import Image
from metric import get_metric
import math
from metric import process
# import jax
# from jax import random as jrandom
# from jax import numpy as jnp
# from jax import vmap as jvmap
# import numpy as np
import gc


def get(set, x):
    return set[x]
get_I_subset1 = torch.vmap( get, in_dims=(None, 0)) # for set: (n_samples , *I_dim) and x: (n_units, N), return: (n_units, N, *I_dim)
get_I_subset2 = torch.vmap(get_I_subset1, in_dims=(None, 0)) # for set: (n_samples , *I_dim) and x: (n_units, N, K+1), return: (n_units, N, K+1, *I_dim)
get_act_subset = torch.vmap( torch.vmap(get, in_dims=(None, 0)) ) # for set: (n_units, n_samples) and x: (n_units, N, K+1), return: (n_units, N, K+1, n_samples)
get_v = torch.vmap(get)
get_vv = torch.vmap(get_v) # for set: (n_units, N, L) and x: (n_units, N, K+1), return: (n_units, N, K+1)

torch_draw_k = torch.vmap(lambda x, L, k: torch.randperm(L)[:k], 
                          in_dims=(0, None, None), randomness='different', chunk_size=4)
torch_draw_k_batch = torch.vmap(torch_draw_k, in_dims=(0, None, None), randomness='different', chunk_size=4)

'''jdraw_k = jvmap(lambda key, L, k: jrandom.choice(key, L, shape=(k,), replace=False),
               in_axes=(0, None, None) )
jdraw_k_batch_v = jvmap(jdraw_k, in_axes=(0, None, None))

def jdraw_k_batch(keys_set, L, k):
    return jnp.stack([jdraw_k(keys, L, k) for keys in keys_set], axis=0)'''


def subset_sampling(seed: int, activations, K: int, N: int, 
                    quantile: float | int, device, 
                    activations_sort_id=None):
    """
    Randomly generates indices of query and explanation images from appropriate quantile range
    for ensemble of tasks. 

    Input:
    - 'seed': seed for random generation
    - 'activations': (n_units x ds_length) activations in different units across all sample images
    - 'K': Number of images in each explanation
    - 'N': Number of tasks
    - 'quantile': The quantile range to draw from
    - 'device': the device to perform random generation on. (Not implemented)
    - 'activations_sort_id': Optional. Argsort of the activations for all units

    Output:
    - 'top_id': (n_units x N x (K+1)) indices of the 
    """


    n_units = activations.shape[0]
    n_samples = activations.shape[1]
    subset_length = math.ceil(n_samples * quantile)
    assert not subset_length < K+1

    torch.manual_seed(seed)

    # sampling without replacement
    top_id = torch_draw_k_batch(torch.empty(n_units, N), subset_length, K+1) 
    bottom_id = torch_draw_k_batch(torch.empty(n_units, N), subset_length, K+1)

    # sampling with replacement
    # top_id = torch.randint(0, subset_length, size=(n_units, N, K+1))
    # bottom_id = torch.randint(0, subset_length, size=(n_units, N, K+1))

    if quantile==1:
        del activations
        return top_id , bottom_id

    if activations_sort_id is None:
        activations_sort_id = torch.argsort(activations, dim=-1, descending=False)
        del activations

    top_set_id = torch.flip(activations_sort_id, [-1])[:, :subset_length]
    bottom_set_id = activations_sort_id[:, :subset_length]
    del activations_sort_id
    
    top_id = get_act_subset(top_set_id , top_id)
    del top_set_id
    bottom_id = get_act_subset(bottom_set_id , bottom_id)
    del bottom_set_id

    torch.cuda.empty_cache()
    return top_id , bottom_id

def sort_subset_id(top_id, bottom_id, activations):
    """
    Sort subsets of indices based on their activation values.

    Args:
        top_id: Tensor of indices for top activations (N_units x N x K+1)
        bottom_id: Tensor of indices for bottom activations (N_units x N x K+1) 
        activations: Tensor of activation values (N_units x N_images)

    Returns:
        tuple: (top_id_sorted, bottom_id_sorted)
            - top_id_sorted: Indices sorted by decreasing activation values
            - bottom_id_sorted: Indices sorted by increasing activation values
    """
    top_act = get_act_subset(activations, top_id)
    bottom_act = get_act_subset(activations, bottom_id)
    assert top_id.shape == bottom_id.shape
    assert top_act.shape == top_id.shape
    assert top_act.shape == bottom_act.shape

    top_sort_id = torch.argsort(top_act, dim=-1, descending=True)
    bottom_sort_id = torch.argsort(bottom_act, dim=-1, descending=False)

    del top_act
    del bottom_act

    top_id_sorted = get_vv(top_id, top_sort_id)
    del top_id
    bottom_id_sorted = get_vv(bottom_id, bottom_sort_id)
    del bottom_id

    del top_sort_id
    del bottom_sort_id

    torch.cuda.empty_cache()
    return top_id_sorted, bottom_id_sorted


def query_explanation_generation(seed: int, I_set, activations, 
                                 K: int, N: int, 
                                 quantile: float,
                                 device,
                                 activations_sort_id=None):
    """
    Generate query and explanations for ALL psychophysics tasks.

    Input:
    - 'I_set': Torch tensor of preprocessed image dataset. Its first 
               dimension is length of the dataset, while subsequent dimensions depend on
               the model-specific input processing. For example, with Dreamsim
               preprocessing which converts a (WxHx3) image to a (1x1792) embedding, the
               dimensions of corresponding 'I_set' will be (N_images x 1792).
               For LPIPS preprocess which recasts a (WxHx3) image to a (3xWxH) tensor,
               the tensor dimensions of corresponding 'I_set' will be (N_images x 3 x W x H).

    - 'activations': A two dimensional tensor (N_units x N_images ) that
                     contains activations of every image along every unit.

    - 'K': The number of images in each (+ , -) explanation set per task.
    - 'N': Number of tasks per unit.
    - 'quantile': A scalar between 0 and 1. The quantile range over the activations for pschophysics query/explanation generation

    Internal:
    - 'I_dim': Dimensions of a single preprocessed image sample

    Output:
    - 'query_set': A tuple ('query_plus_set', 'query_minus_set') containing batched queries for all psychophysics tasks.
        - 'query_plus_set' , 'query_minus_set': torch tensor of shape (N_units, N, *I_dim)

    - 'Explanation_set': A tuple ('Explanation_plus_set', 'Explanation_minus_set') containing batched explanations for all psychophysics tasks.
        - 'Explanation_plus_set', 'Explanation_minus_set': torch tensor of shape (N_units, N, K, *I_dim)
    """
    top_id , bottom_id = subset_sampling(seed, activations, K=K, 
                                         N=N, quantile=quantile, device=device,
                                         activations_sort_id=activations_sort_id)
    
    top_id , bottom_id = sort_subset_id(top_id, bottom_id, activations)

    Explanation_plus_set = get_I_subset2(I_set, top_id[:,:,:K])
    query_plus_set = get_I_subset1(I_set, top_id[:,:,K])
    del top_id

    Explanation_minus_set = get_I_subset2(I_set, bottom_id[:,:,:K])
    query_minus_set = get_I_subset1(I_set, bottom_id[:,:,K])
    del bottom_id

    query_set = (query_plus_set.to(device), query_minus_set.to(device))
    del query_plus_set
    del query_minus_set

    Explanation_set = (Explanation_plus_set.to(device), Explanation_minus_set.to(device))
    del Explanation_plus_set
    del Explanation_minus_set

    torch.cuda.empty_cache()
    return query_set , Explanation_set


def aggregate(im_sim):
    """
    Aggregate similarity scores across all tasks.
    """
    a = torch.mean(im_sim, dim=-1)
    if a.shape[0]==1 and len(a.shape)==1:
        return a[0]
    return a
aggregate_batch = torch.vmap(aggregate)


def s(q_batch, E_batch, sim_metric_v):
    """
    Calculate similarity between query and explanation images.

    Input:
    - 'q_batch': Torch tensor of shape (N, *I_dim) containing preprocessed query images
    - 'E_batch': Torch tensor of shape (N, K, *I_dim) containing preprocessed explanation images
    - 'sim_metric_v': Callable similarity metric function that is vectorized with torch vmap

    Output:
    - 'a_batch': Torch tensor of shape (N,) containing similarity scores
    """
    sim_batch = sim_metric_v(q_batch, E_batch) # similarity should be inverse relationship to distance metric
    a_batch = aggregate_batch(sim_batch)

    return a_batch


def calc_MIS(query, Explanation, sim_metric_v: callable, alpha: float|None=None):
    """
    Calculate Mechanistic Interpretability Score (MIS) for SINGLE UNIT

    Input:
    - 'query': A tuple ('q_plus', 'q_minus') containing queries of all psychophysics tasks for SINGLE UNIT.
        - 'q_plus', 'q_minus': Torch tensor of shape (N , *I_dim)

    - 'Explanation': A tuple ('E_plus' , 'E_minus') containing explanations of all psychophysics tasks for SINGLE UNIT.
        - 'E_plus', 'E_minus': Torch tensor of shape (N , K, *I_dim)

    - 'sim_metric_v': Callable similarity metric function that is vectorized with torch vmap

    - 'alpha': Parametre for Sigmoid function in MIS calculation. If None, defaults to unnormalized psychophysics accuracy.

    Output:
    - 'MIS': Torch scalar of MIS of SINGLE UNIT

    """

    E_plus , E_minus = Explanation
    q_plus , q_minus = query

    assert q_plus.shape[0]==E_plus.shape[0]
    assert q_minus.shape[0]==E_minus.shape[0]
    assert q_plus.shape[0]==q_minus.shape[0]

    s_plus_plus = s(q_plus, E_plus, sim_metric_v)
    s_plus_minus = s(q_plus, E_minus, sim_metric_v)
    s_minus_plus = s(q_minus, E_plus, sim_metric_v)
    s_minus_minus = s(q_minus, E_minus, sim_metric_v)

    delta_plus = s_plus_plus - s_plus_minus
    delta_minus = s_minus_plus - s_minus_minus
    delta_difference = delta_plus - delta_minus

    if alpha is None:
        MIS = torch.sum(delta_difference > 0, dim=-1) / len(delta_difference)
        return MIS

    MIS = torch.sigmoid(alpha * delta_difference )
    MIS = torch.mean(MIS)
    return MIS


def calc_MIS_set(query_set, Explanation_set, sim_metric: callable, alpha=0.16):
    """
    Wrapper for callable 'calc_MIS' to compute MIS of ALL UNITS

    Input:
    - 'query_set': A tuple ('q_plus_set', 'q_minus_set') containing query processed images of every psychophysics task for ALL UNITS.
        - 'q_plus_set', 'q_minus_set': Torch tensor of shape (N_units, N, *I_dim)

    - 'Explanation_set': A tuple ('Explanation_plus_set', 'Explanation_minus_set') containing explanation processed images
                         of every psychophysics task for ALL UNITS.
        - 'Explanation_plus_set', 'Explanation_minus_set': Torch tensor of shape (N_units, N, K, *I_dim)

    - 'sim_metric': Callable similarity metric function to be passed into callable 'calc_MIS'

    - 'alpha': Parametre for Sigmoid function in MIS calculation, passed into callable 'calc_MIS'

    Output:
    - 'MIS_set': Torch tensor of shape (N_units,). Contains MIS of ALL UNITS.
    """

    assert len(query_set)==2
    assert len(Explanation_set)==2
    assert query_set[0].shape[0]==Explanation_set[0].shape[0]
    assert query_set[1].shape[0]==Explanation_set[1].shape[0]
    assert query_set[0].shape[0]==query_set[1].shape[0]

    (query_plus_set , query_minus_set) = query_set
    (Explanation_plus_set , Explanation_minus_set) = Explanation_set

    query_set = [ (q_plus , q_minus) for (q_plus , q_minus) in zip(query_plus_set , query_minus_set) ]
    Explanation_set = [ (E_plus , E_minus) for (E_plus , E_minus) in zip(Explanation_plus_set , Explanation_minus_set) ]

    MIS_set = torch.tensor([
        calc_MIS(query, Explanation, torch.vmap(sim_metric), alpha)
        for (query , Explanation) in zip(query_set , Explanation_set)
    ])

    return MIS_set


class task_config:
    """
    A class to store and manage task data including images, activations, and preprocessed versions.
    
    Attributes:
        x_data: List of PIL Images (N_images x H x W x C) representing the input images
        y_data: Torch tensor (N_units x N_images) containing activations, transposed from (N_images x N_units)
        y_sort_id: Sorted indices of y_data in ascending order
        device: String specifying the compute device ('cpu' or 'cuda')
        processed: Dictionary storing preprocessed versions of x_data for different metrics
    """

    def __init__(self, device: str, image_set: list[Image], activations: Tensor,
                 processed: dict={}):
        '''
        Initialize task_config with images and their corresponding activations.

        Args:
            device: String specifying compute device ('cpu' or 'cuda')
            image_set: List of PIL Images (N_images x H x W x C)
            activations: Torch tensor (N_images x N_units) containing activation values
            processed: Optional dict of preprocessed versions of image_set
        '''

        self.x_data = image_set
        self.y_data = torch.transpose(activations , 0, 1)
        self.y_sort_id = torch.argsort(self.y_data, dim=1, descending=False)
        self.device = device
        self.processed = processed

    def __getitem__(self, index):
        """Get image and activation pair at given index."""
        return self.x_data[index] , self.y_data[:,index]
    
    def get_data(self, metric_type: str=None):
        """
        Get raw or preprocessed image data for specified metric type.
        
        Args:
            metric_type: String specifying which preprocessed version to return.
                        If None, returns raw images.
        
        Returns:
            Raw or preprocessed image data
        """
        if metric_type is None:
            return self.x_data
        if metric_type in self.processed.keys():
            return self.processed[metric_type]
        
        self.processed[metric_type] = process(self.x_data, metric_type, self.device)
        return self.processed[metric_type]
    
    def get_target(self):
        """Get activation values."""
        return self.y_data
    
    def replace_y(self, activations):
        """
        Replace current activation values with new ones.
        
        Args:
            activations: New activation tensor (N_images x N_units)
        """
        self.y_data = torch.transpose(activations , 0, 1)
        self.y_sort_id = torch.argsort(self.y_data, dim=1, descending=False)


def run_psychophysics(seed: int, task_data: task_config, metric_type: str,
                      K: int, N: int, quantile: float, alpha: float=None, metric=None):
    """
    Run psychophysics experiment to generate Mechanistic Interpretability Score (MIS).
    
    Args:
        seed: Random seed for reproducibility
        task_data: Task configuration object containing image data and activations
        metric_type: Type of similarity metric to use ('dreamsim' or 'lpips')
        K: Number of images in each (+ , -) explanation set per task
        N: Number of psychophysics experiment trials
        quantile: Quantile threshold for selecting images
        alpha: Optional threshold parameter for MIS calculation
        metric: Optional pre-initialized metric function. If None, will be initialized based on metric_type
    
    Returns:
        MIS_set: Set of Mechanistic Interpretability Score calculated through psychophysics experiment
    """
    
    device = task_data.device

    if metric is None:
        metric = get_metric(metric_type, task_data.device)

    query_set , Explanation_set = \
        query_explanation_generation(seed=seed, I_set=task_data.get_data(metric_type=metric_type), 
                                     activations=task_data.y_data, K=K, N=N,
                                     quantile=quantile, device=device,
                                     activations_sort_id=task_data.y_sort_id)
    
    MIS_set = calc_MIS_set(query_set, Explanation_set, metric, alpha=alpha)
    del query_set
    del Explanation_set

    return MIS_set