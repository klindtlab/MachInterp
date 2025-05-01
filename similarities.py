'''
- Purpose: to load metrics (e.g., lpips, dreamsim, etc.) with precomputed image similarity matrices
- In order, this:
1) Checks for the existence of precomputed similarity matrices in shared + local folders
2) If they exist: load metric with precomputed matrices (fast)
3) If they do not: precomputes then loads the metrics + saves newly precomputed matrices in shared folder if possible, saves in local if not (slow)

Example use:
import similarities
from similarities import loadMetrics
sim_metrics = ['dreamsim','lpips']
all_metrics=loadMetrics(sim_metrics=sim_metrics, identifier=identifier, region=region, images=images)
dreamsim = all_metrics['dreamsim']
lpips = all_metrics['lpips']

'''
# Imports
import os
import metric
import numpy as np

# Default local folder is ./precomputed
local_dir = 'precomputed' 
os.makedirs(local_dir, exist_ok=True)


# Metric 1: Dreamsim
def loadDreamsim(identifier, region, images, sim_metric='dreamsim'):
    # Specify desired shared folder, create paths
    shared_dir = f'/grid/klindt/data/Neuro/{identifier}/precomputed'
    file = f'dreamsim_precomputed_{identifier}_{region}.npy'
    local_path = os.path.join(local_dir, file)
    shared_path = os.path.join(shared_dir, file)
    # Search for precomputed embeddings, otherwise precompute and save
    if os.path.exists(shared_path):
        dreamsim = metric.PrecomputedMetric(np.load(shared_path)) #load embeddings if in shared folder
        print(f'{sim_metric} embeddings found in shared folder')
    elif os.path.exists(local_path):
        dreamsim = metric.PrecomputedMetric(np.load(local_path)) #load embeddings if in local folder
        print(f'{sim_metric} embeddings found in local folder')
    else:
        print (f'{sim_metric} embeddings not found, precomputing now...' )
        dreamsim = metric.DreamSimMetric() # get metric: dreamsim
        dreamsim.precompute(images, batch_size=256) #precompute embeddings
        try:
            np.save(shared_path, dreamsim.similarity_matrix) #save embeddings to shared folder if possible
            print(f'Embeddings for {sim_metric} saved to shared folder.')
        except PermissionError:
            np.save(local_path, dreamsim.similarity_matrix) #save embeddings to local folder if permission error
            print(f'Shared folder permission denied - Embeddings for {sim_metric} saved to local folder: ./{local_dir}')
    print(dreamsim.similarity_matrix.shape)
    return dreamsim
    
# Metric 2: LPIPS
def loadLPIPS(identifier, region, images, sim_metric='lpips'):
    # Specify desired shared folder, create paths
    shared_dir = f'/grid/klindt/data/Neuro/{identifier}/precomputed'
    file = f'lpips_precomputed_{identifier}_{region}.npy'
    local_path = os.path.join(local_dir, file)
    shared_path = os.path.join(shared_dir, file)
    # Search for precomputed embeddings, otherwise precompute and save
    if os.path.exists(shared_path):
        lpips = metric.PrecomputedMetric(np.load(shared_path)) #load embeddings if in shared folder
        print(f'{sim_metric} embeddings found in shared folder')
    elif os.path.exists(local_path):
        lpips = metric.PrecomputedMetric(np.load(local_path)) #load embeddings if in local folder
        print(f'{sim_metric} embeddings found in local folder')
    else:
        print (f'{sim_metric} embeddings not found, precomputing now...' )
        lpips = metric.LPIPSMetric() # get metric: lpips
        lpips.precompute(images, batch_size=64) # precompute embeddings
        try:
            np.save(shared_path, lpips.similarity_matrix) #save embeddings to shared folder if possible
            print(f'Embeddings for {sim_metric} saved to shared folder.')
        except PermissionError:
            np.save(local_path, lpips.similarity_matrix) #save embeddings to local folder if permission error
            print(f'Shared folder permission denied - Embeddings for {sim_metric} saved to local folder: ./{local_dir}')
    print(lpips.similarity_matrix.shape)
    return lpips
    

def loadMetrics(sim_metrics, identifier, region, images):
    all_metrics = {}
    for sim_metric in sim_metrics:
        if sim_metric == 'dreamsim':
            dreamsim = loadDreamsim(identifier=identifier, region=region, images=images, sim_metric='dreamsim')
            all_metrics['dreamsim'] = dreamsim
        elif sim_metric == 'lpips':
            lpips = loadLPIPS(identifier=identifier, region=region, images=images, sim_metric='lpips')
            all_metrics['lpips'] = lpips
        else:
            print(f'Invalid metric: {sim_metric}')
    return all_metrics