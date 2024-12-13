import torch

def single_shuffle(t):
    shuffle_id = torch.randperm(t.shape[-1])
    return shuffle_id, t[shuffle_id]


def subset_sampling(activations, K: int, N: int, quantile: float | int):
    if quantile == 0 or quantile is None:
        top_q = 0
        bott_q = 1
    else:
        top_q = max(quantile, 1-quantile)
        bott_q = min(quantile, 1-quantile)
    floor = torch.quantile(activations, q=top_q, dim=-1)
    ceil = torch.quantile(activations, q=bott_q, dim=-1)

    def single_sampling_top(act_i, floor_i):
        sub_id = torch.where(act_i >= floor_i)[0]
        sub_act = act_i[sub_id]
        single_shuffle_v = torch.vmap(lambda x: single_shuffle(sub_act)[:K+1], randomness="different")
        output_id, output_act = single_shuffle_v(torch.empty(N))
        return output_id, output_act

    def single_sampling_bottom(act_i, ceil_i):
        sub_id = torch.where(act_i <= ceil_i)[0]
        sub_act = act_i[sub_id]
        single_shuffle_v = torch.vmap(lambda x: single_shuffle(sub_act)[:K+1], randomness="different")
        output_id, output_act = single_shuffle_v(torch.empty(N))
        return output_id, output_act

    top_id, top_subset = torch.vmap(single_sampling_top)(activations, floor)
    bottom_id, bottom_subset = torch.vmap(single_sampling_bottom)(activations, ceil)

    return (top_id, bottom_id), (top_subset, bottom_subset)

def assign(t, id):
    return t[id]
assign_vv = torch.vmap(torch.vmap(assign))

def sort_top_bottom_id(top_bottom_id, top_bottom_activations):
    (top_id , bottom_id) = top_bottom_id
    (top_activations , bottom_activations) = top_bottom_activations

    assert top_id.shape[1] == bottom_id.shape[1]
    N = top_id.shape[1]

    top_sort_id = torch.argsort(top_activations, dim=-1, descending=True)
    bottom_sort_id = torch.argsort(bottom_activations, dim=-1, descending=False)

    top_id = assign_vv(top_id, top_sort_id)
    bottom_id = assign_vv(bottom_id, bottom_sort_id)

    return top_id, bottom_id


def query_explanation_generation(I_set, activations, K: int=9, N: int=20, quantile: float=0.2):
    """
    Generate query and explanations for ALL psychophysics tasks.

    Input:
    - 'I_set': Torch tensor of preprocessed image dataset. Its first 
               dimension is length of the dataset, while subsequent dimensions depend on
               the model-specific input processing. For example, with Dreamsim
               preprocessing which converts a (WxHx3) image to a (1x1792) embedding, the
               dimensions of corresponding 'I_set' will be (ds_length x 1792).
               For LPIPS preprocess which recasts a (WxHx3) image to a (3xWxH) tensor,
               the tensor dimensions of corresponding 'I_set' will be (ds_length x W x H x 3).

    - 'activations': A two dimensional tensor (n_units x ds_length ) that
                     contains activations of every image along every unit.

    - 'K': The number of images in each (+ , -) explanation set per task.
    - 'N': Number of tasks per unit.
    - 'quantile': A scalar between 0 and 1. The quantile range over the activations for pschophysics query/explanation generation

    Internal:
    - 'I_dim': Dimensions of a single preprocessed image sample

    Output:
    - 'query_set': A tuple ('query_plus_set', 'query_minus_set') containing batched queries for all psychophysics tasks.
        - 'query_plus_set' , 'query_minus_set': torch tensor of shape (n_units, N, *I_dim)

    - 'Explanation_set': A tuple ('Explanation_plus_set', 'Explanation_minus_set') containing batched explanations for all psychophysics tasks.
        - 'Explanation_plus_set', 'Explanation_minus_set': torch tensor of shape (n_units, N, K, *I_dim)
    """

    n_units = activations.shape[0]
    I_dim = I_set.shape[1:]

    Explanation_plus_set = torch.zeros(n_units, N, K, *I_dim)
    Explanation_minus_set = torch.zeros(n_units, N, K, *I_dim)

    query_plus_set = torch.zeros(n_units, N, *I_dim)
    query_minus_set = torch.zeros(n_units, N, *I_dim)

    top_bottom_id, top_bottom_activations = subset_sampling(activations, K=K, N=N, quantile=quantile)
    top_id , bottom_id = sort_top_bottom_id(top_bottom_id, top_bottom_activations)

    for ii in range(n_units):
        for jj in range(N):
            Explanation_plus_set[ii,jj] = I_set[ top_id[ii,jj,:K] ]
            Explanation_minus_set[ii,jj] = I_set[ bottom_id[ii,jj,:K] ]

            query_plus_set[ii,jj] = I_set[ top_id[ii,jj,K] ]
            query_minus_set[ii,jj] = I_set[ bottom_id[ii,jj,K] ]

    query_set = (query_plus_set, query_minus_set)
    Explanation_set = (Explanation_plus_set, Explanation_minus_set)

    return query_set , Explanation_set


def aggregate(im_sim):
    a = torch.mean(im_sim, dim=-1)
    if a.shape[0]==1 and len(a.shape)==1:
        return a[0]
    return a
aggregate_batch = torch.vmap(aggregate)


def s(q_batch, E_batch, sim_metric_v):
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

    - 'sim_metric': Callable similarity metric function.

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
        - 'q_plus_set', 'q_minus_set': Torch tensor of shape (n_units, N , *I_dim)

    - 'Explanation_set': A tuple ('Explanation_plus_set', 'Explanation_minus_set') containing explanation processed images
                         of every psychophysics task for ALL UNITS.
        - 'Explanation_plus_set', 'Explanation_minus_set': Torch tensor of shape (n_units, N, K, *I_dim)

    - 'sim_metric': Callable similarity metric function to be passed into callable 'calc_MIS'

    - 'alpha': Parametre for Sigmoid function in MIS calculation, passed into callable 'calc_MIS'

    Output:
    - 'MIS_set': Torch tensor of shape (n_units,). Contains MIS of ALL UNITS.
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