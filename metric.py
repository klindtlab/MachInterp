import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class Metric:
    """
    Abstract base class for different metrics.
    
    Subclasses must implement:
      - preprocess: convert raw inputs (e.g., numpy arrays, torch tensors, list of file paths)
                     into a common format for similarity computation.
      - similarity: compute a pairwise similarity matrix between two batches.
    """
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.precomputed = False

    def _to_tensor(self, inputs):
        if isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs)
        elif torch.is_tensor(inputs):
            return inputs
        else:
            raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")
    
    def _to_numpy(self, inputs):
        if isinstance(inputs, np.ndarray):
            return inputs
        elif torch.is_tensor(inputs):
            return inputs.cpu().numpy()
        else:
            raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")

    def preprocess(self, inputs):
        raise NotImplementedError('Subclasses must implement preprocess')

    def similarity(self, batch_A, batch_B) -> np.ndarray:
        raise NotImplementedError('Subclasses must implement similarity')

    def compute_similarity(self, batch_A, batch_B) -> np.ndarray:
        """
        Convenience method that wraps the similarity computation.
        """
        if self.precomputed:
            pre_A = batch_A
            pre_B = batch_B
        else:
            try:
                pre_A = self.preprocess(batch_A)
                pre_B = self.preprocess(batch_B)
            except Exception as e:
                raise ValueError(f"Error during preprocessing: {e}")
        try:
            sim = self.similarity(pre_A, pre_B)
        except Exception as e:
            raise ValueError(f"Error during similarity computation: {e}")
        return sim
    
    def precompute(self, inputs, batch_size):
        raise NotImplementedError('Subclass has not implemented precompute')
    
    def precomputed_similarity(self, ind_batch_A, ind_batch_B):
        raise NotImplementedError('Subclass has not implemented precomputed_similarity')


class DreamSimMetric(Metric):
    def __init__(self, device: str = 'cpu'):
        super().__init__(device)
        from dreamsim import dreamsim
        self.model, _ = dreamsim(pretrained=True, device=device)
        self.model.eval()
        self.embed_fn = self.model.embed
        self.precomputed = False

    def preprocess(self, inputs):
        tensor = self._to_tensor(inputs)
        return tensor.to(self.device, torch.float32)

    def similarity(self, batch_A, batch_B) -> np.ndarray:
        """
        Compute cosine similarity between embeddings.
        Converts numpy arrays to torch tensors if needed.
        """
        with torch.no_grad():
            embed_A = self.embed_fn(batch_A)
            embed_B = self.embed_fn(batch_B)
            similarity_matrix = F.cosine_similarity(embed_A[:, None], embed_B[None], dim=-1)
        return similarity_matrix.cpu().numpy()
    
    def precompute(self, inputs, batch_size: int = 256):
        """
        Precompute embeddings for all inputs.
        batch_size is just for precomputing of embeddings.
        """
        print('Dreamsim: Embedding images')
        dataloader = torch.utils.data.DataLoader(
            TensorDataset(self._to_tensor(inputs).to(torch.float32)),
            batch_size=batch_size, 
            shuffle=False
        )
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                embeddings.append(self.embed_fn(batch[0].to(self.device)).cpu())
        self.embeddings = torch.cat(embeddings, dim=0)
        assert self.embeddings.shape[0] == len(inputs), "Embedding shape does not match input length"
        print('Dreamsim: Precomputed embeddings. Now, use precomputed_similarity with indices!')
        self.precomputed = True

    def precomputed_similarity(self, ind_batch_A, ind_batch_B):
        assert self.precomputed, "Precomputed embeddings are not available"
        assert len(ind_batch_A.shape) == 1 and len(ind_batch_B.shape) == 1, "inputs must be 1D arrays of indices"
        ind_batch_A = self._to_tensor(ind_batch_A).to(torch.int64)
        ind_batch_B = self._to_tensor(ind_batch_B).to(torch.int64)
        with torch.no_grad():
            embed_A = self.embeddings[ind_batch_A].to(self.device)
            embed_B = self.embeddings[ind_batch_B].to(self.device)
            similarity_matrix = F.cosine_similarity(embed_A[:, None], embed_B[None], dim=-1)
        return similarity_matrix.cpu().numpy()


class LPIPSMetric(Metric):
    def __init__(self, device: str = 'cpu', ret_per_layer: bool = False):
        super().__init__(device)
        # use custom lpips version with batch support
        # https://github.com/david-klindt/PerceptualSimilarity/tree/batched
        from lpips import LPIPS
        self.loss_fn = LPIPS(net='alex').to(self.device)
        self.ret_per_layer = ret_per_layer

    def preprocess(self, inputs):
        assert inputs.min() >= -1 and inputs.max() <= 1, "Input images must be normalized to [-1, 1]"
        tensor = self._to_tensor(inputs)
        return tensor.to(self.device, torch.float32)
    
    def similarity(self, batch_A, batch_B) -> np.ndarray:
        """
        Compute LPIPS similarity between every pair in the two batches.
        Returns negative LPIPS distance as similarity.
        """
        with torch.no_grad():
            output = self.loss_fn(batch_A, batch_B, normalize=True, retPerLayer=self.ret_per_layer)
            if self.ret_per_layer:
                output = torch.stack(output[1], dim=-1)
        return - output.detach().cpu().numpy()


class SSIMMetric(Metric):
    def __init__(self):
        super().__init__()
        from skimage.metrics import structural_similarity as ssim_func
        self.ssim_func = ssim_func

    def preprocess(self, inputs):
        assert inputs.min() >= -1 and inputs.max() <= 1, "Input images must be normalized to [-1, 1]"
        return self._to_numpy(inputs)

    def similarity(self, batch_A, batch_B) -> np.ndarray:
        """
        Compute SSIM (Structural Similarity Index) between every pair.
        Converts images to grayscale if necessary.
        """
        num_A = len(batch_A)
        num_B = len(batch_B)
        similarity_matrix = np.zeros((num_A, num_B))
        for i in range(num_A):
            for j in range(num_B):
                similarity_matrix[i, j] = self.ssim_func(
                    batch_A[i], batch_B[j], channel_axis=0, data_range=2)
        return similarity_matrix
    

def flatten_images(images):
    """
    Flatten images to 2D arrays.
    """
    return np.reshape(images, (images.shape[0], -1))


class MSEMetric(Metric):
    def __init__(self):
        super().__init__()

    def preprocess(self, inputs):
        return self._to_numpy(inputs)

    def similarity(self, batch_A, batch_B) -> np.ndarray:
        """
        Compute negative mean squared error between every pair.
        """
        flat_A = flatten_images(batch_A)
        flat_B = flatten_images(batch_B)
        # Compute pairwise differences using broadcasting.
        mse = np.mean((flat_A[:, None, :] - flat_B[None, :, :]) ** 2, axis=-1)
        return -mse
    

class CosineMetric(Metric):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def preprocess(self, inputs):
        return self._to_numpy(inputs)

    def similarity(self, batch_A, batch_B) -> np.ndarray:
        flat_A = flatten_images(batch_A)
        flat_B = flatten_images(batch_B)
        A_norm = flat_A / (np.linalg.norm(flat_A, axis=1, keepdims=True) + self.epsilon)
        B_norm = flat_B / (np.linalg.norm(flat_B, axis=1, keepdims=True) + self.epsilon)
        return np.dot(A_norm, B_norm.T)


class LabelMetric(Metric):
    def __init__(self):
        super().__init__()

    def preprocess(self, inputs):
        return self._to_numpy(inputs)

    def similarity(self, batch_A, batch_B) -> np.ndarray:
        """
        Compute similarity based on equality: 1 if labels are equal, 0 otherwise.
        """
        return (np.array(batch_A)[:, None] == np.array(batch_B)[None, :]).astype(float)
