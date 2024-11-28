import torch

def random_shuffle(t, N):
    if len(t.shape)==2:
        t = t.squeeze(0)

    t_shuffled = torch.zeros(N, t.shape[-1], dtype=int)
    for ii in range(N):
        t_shuffled[ii] = t[torch.randperm(t.shape[-1])]

    return t_shuffled


def subset_sampling(activations, K: int, N: int, quantile: float | int):
    n_units = activations.shape[0]
    if quantile == 0:
        top_q = 0
        bott_q = 1
    else:
        top_q = max(quantile, 1-quantile)
        bott_q = min(quantile, 1-quantile)
    floor = torch.quantile(activations, q=top_q, dim=-1)
    ceil = torch.quantile(activations, q=bott_q, dim=-1)

    top_subset_id = []
    bottom_subset_id = []
    for ii in range(n_units):
        top_subset_id.append(
            torch.where(activations[ii] >= floor[ii])[0]
        )
        bottom_subset_id.append(
            torch.where(activations[ii] <= ceil[ii])[0]
        )

        assert not (len(top_subset_id[ii]) < K+1 )
        assert not (len(bottom_subset_id[ii]) < K+1 )

    top_id = torch.zeros(n_units, N,  K+1, dtype=int)
    bottom_id = torch.zeros(n_units, N,  K+1, dtype=int)
    for ii in range(n_units):
        top_id[ii] = random_shuffle(top_subset_id[ii], N)[:,:K+1]
        bottom_id[ii] = random_shuffle(bottom_subset_id[ii], N)[:,:K+1]

    return top_id , bottom_id


def sort_top_bottom_id(activations, top_id, bottom_id):
    assert top_id.shape[1]==bottom_id.shape[1]
    N = top_id.shape[1]
    n_units = activations.shape[0]

    top_subset = torch.zeros_like(top_id)
    bottom_subset = torch.zeros_like(bottom_id)

    for ii in range(n_units):
        for jj in range(N):
            top_subset[ii,jj] = activations[ii, top_id[ii,jj]]
            bottom_subset[ii,jj] = activations[ii, bottom_id[ii,jj]]

    top_sort_id = torch.argsort(top_subset, dim=-1, descending=True)
    bottom_sort_id = torch.argsort(bottom_subset, dim=-1, descending=False)

    for ii in range(n_units):
        for jj in range(N):
            top_id[ii,jj] = top_id[ii, jj, top_sort_id[ii,jj] ]
            bottom_id[ii,jj] = bottom_id[ii, jj, bottom_sort_id[ii,jj] ]

    return top_id, bottom_id


def query_explanation_generation(I_set, activations, K: int=9, N: int=20, quantile: float=0.2):
    n_units = activations.shape[0]
    n_dim = I_set.shape[1]

    Explanation_plus_set = torch.zeros(n_units, N, K, n_dim)
    Explanation_minus_set = torch.zeros(n_units, N, K, n_dim)

    query_plus_set = torch.zeros(n_units, N, n_dim)
    query_minus_set = torch.zeros(n_units, N, n_dim)

    top_id , bottom_id = subset_sampling(activations, K=K, N=N, quantile=quantile)
    top_id , bottom_id = sort_top_bottom_id(activations, top_id, bottom_id)

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


def s(q, E, sim_metric):
    sim = sim_metric(q,E) # similarity should be inverse relationship to distance metric
    a = aggregate(sim)

    return a


def calc_MIS(query, Explanation, sim_metric: callable, alpha=None):
    E_plus , E_minus = Explanation
    q_plus , q_minus = query

    assert q_plus.shape[0]==E_plus.shape[0]
    assert q_minus.shape[0]==E_minus.shape[0]
    assert q_plus.shape[0]==q_minus.shape[0]

    s_plus_plus = torch.tensor([s(q, E, sim_metric) for q , E in zip(q_plus,E_plus) ])
    s_plus_minus = torch.tensor([s(q, E, sim_metric) for q , E in zip(q_plus,E_minus) ])
    s_minus_plus = torch.tensor([s(q, E, sim_metric) for q , E in zip(q_minus,E_plus) ])
    s_minus_minus = torch.tensor([s(q, E, sim_metric) for q , E in zip(q_minus,E_minus) ])

    delta_plus = s_plus_plus - s_plus_minus
    delta_minus = s_minus_plus - s_minus_minus
    delta_difference = delta_plus - delta_minus

    if alpha is None:
        MIS = delta_difference > 0
        MIS = MIS / len(MIS)
        return MIS

    MIS = torch.sigmoid(alpha * delta_difference )
    MIS = torch.mean(MIS)
    return MIS


def calc_MIS_set(query_set, Explanation_set, sim_metric: callable, alpha=0.16):
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
        calc_MIS(query, Explanation, sim_metric, alpha)
        for (query , Explanation) in zip(query_set , Explanation_set)
    ])

    return MIS_set