import torch

def get_dreamsim(device):
    from dreamsim import dreamsim

    model, preprocess = dreamsim(pretrained=True, device=device)

    embed_fn = model.embed

    def sim_metric(embed1, embed2):
        if len(embed1.shape)==1:
            embed1 = embed1.unsqueeze(0)
        if len(embed2.shape)==1:
            embed2 = embed2.unsqueeze(0)

        return F.cosine_similarity(embed1[:, None], embed2[None], dim=-1)

    def preprocess_embed_ds(ds, collate_fn: callable=None):
        from torch.utils.data import DataLoader
        from functools import partial
        collate_fn = partial(collate_fn, preprocess=preprocess)
        ds_loader = DataLoader(ds, batch_size=64,
                               shuffle=False, num_workers=2*torch.cuda.device_count(),
                               collate_fn=collate_fn)

        try:
            from tqdm.notebook import tqdm
            loader_loop = tqdm(enumerate(ds_loader), total=len(ds_loader) )
        except ImportError:
            loader_loop = enumerate(ds_loader)

        embedding=[]
        for _ , X in loader_loop:
            embedding.append( embed_fn(X.to(device)) )

        return torch.cat(embedding, dim=0)

    return sim_metric, preprocess_embed_ds

def get_metric(metric_type: str, device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')):
    assert metric_type in ["dreamsim"]
    
    if metric_type=="dreamsim":
        return get_dreamsim(device)
    return None