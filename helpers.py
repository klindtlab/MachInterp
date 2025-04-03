import numpy as np

def flatten_images(images):
    """
    Flatten images to 2D arrays.
    """
    return np.reshape(images, (images.shape[0], -1))

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

def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)

def get_center_ind(activations, quantile):
    """Get indices of all activations in center quantile, only use distribution center between quantiles."""
    return np.logical_and(
        activations >= np.quantile(activations, quantile), 
        activations <= np.quantile(activations, 1 - quantile))

def get_extreme_ind(activations, quantile):
    """Get indices of all activations outside center quantile, only use distribution outside center quantiles."""
    return np.logical_or(
        activations <= np.quantile(activations, quantile), 
        activations >= np.quantile(activations, 1 - quantile))