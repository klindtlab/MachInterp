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
    def __init__(self):
        self.precomputed = False
        self.num_scores = 1  # only relevant for LPIPS with multiple outputs

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
        """Precompute Similarities for entire dataset."""
        N = inputs.shape[0]
        if self.num_scores > 1:
            self.similarity_matrix = np.zeros((N, N, self.num_scores))
        else:
            self.similarity_matrix = np.zeros((N, N))
        for i in tqdm(range(0, N, batch_size)):
            end_i = min(i + batch_size, N)
            x_i = inputs[i:end_i]
            for j in range(0, N, batch_size):
                end_j = min(j + batch_size, N)
                x_j = inputs[j:end_j]
                self.similarity_matrix[i:end_i, j:end_j] = self.compute_similarity(x_i, x_j)
        self.precomputed = True
        print('Precomputed similarities. Now, use precomputed_similarity with indices!')
        print('You can also save metric.similarity_matrix and use it to initialize PrecomputedMetric')
        print('this is to avoid precomputing next time.')
    
    def precomputed_similarity(self, ind_batch_A, ind_batch_B) -> np.ndarray:
        assert self.precomputed, "Precomputed embeddings are not available"
        assert len(ind_batch_A.shape) == 1 and len(ind_batch_B.shape) == 1, "inputs must be 1D arrays of indices"
        ind_batch_A = self._to_numpy(ind_batch_A).astype(int)
        ind_batch_B = self._to_numpy(ind_batch_B).astype(int)
        return self.similarity_matrix[np.ix_(ind_batch_A, ind_batch_B)]
    

class PrecomputedMetric(Metric):
    def __init__(self, similarity_matrix):
        super().__init__()
        shape = similarity_matrix.shape
        assert shape[0] == shape[1], "similarity_matrix must be square."
        self.similarity_matrix = self._to_numpy(similarity_matrix)
        self.precomputed = True
        if len(shape) > 2:
            self.num_scores = shape[2]


class DreamSimMetric(Metric):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
        from dreamsim import dreamsim
        self.model, _ = dreamsim(pretrained=True, device=self.device)
        self.model.eval()
        self.embed_fn = self.model.embed
        self.precomputed = False

    def preprocess(self, inputs):
        tensor = self._to_tensor(inputs)
        return tensor.to(torch.float32)

    def similarity(self, batch_A, batch_B) -> np.ndarray:
        """
        Compute cosine similarity between embeddings.
        Converts numpy arrays to torch tensors if needed.
        """
        with torch.no_grad():
            embed_A = self.embed_fn(batch_A.to(self.device))
            embed_B = self.embed_fn(batch_B.to(self.device))
            similarity_matrix = F.cosine_similarity(embed_A[:, None], embed_B[None], dim=-1)
        return similarity_matrix.cpu().numpy()
    
    def precompute(self, inputs, batch_size: int = 256, embeddings_only: bool = False):
        """
        Precompute embeddings for all inputs.
        The batch_size is just for precomputing of embeddings.
        Use embeddings_only for very large datasets that give OOM.
        """
        print('Embedding images.')
        dataloader = torch.utils.data.DataLoader(
            TensorDataset(self._to_tensor(inputs).to(torch.float32)),
            batch_size=batch_size, 
            shuffle=False
        )
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                embeddings.append(self.embed_fn(batch[0].to(self.device)).cpu())
        self.model.cpu()
        self.embeddings = torch.cat(embeddings, dim=0)
        assert self.embeddings.shape[0] == len(inputs), "Embedding shape does not match input length"
        self.precomputed = True
        self.embeddings_only = embeddings_only
        if not self.embeddings_only:
            print("Computing similarities.")
            self.similarity_matrix = F.cosine_similarity(
                self.embeddings[:, None], self.embeddings[None], dim=-1).numpy()
        print('Dreamsim: Precomputed embeddings and similarities. Now, use precomputed_similarity with indices!')
        print('You can also save metric.similarity_matrix and use it to initialize PrecomputedMetric')
        print('this is to avoid precomputing next time.')

    def precomputed_similarity(self, ind_batch_A, ind_batch_B) -> np.ndarray:
        assert self.precomputed, "Precomputed embeddings are not available"
        assert len(ind_batch_A.shape) == 1 and len(ind_batch_B.shape) == 1, "inputs must be 1D arrays of indices"
        if self.embeddings_only:
            ind_batch_A = self._to_tensor(ind_batch_A).to(torch.int64)
            ind_batch_B = self._to_tensor(ind_batch_B).to(torch.int64)
            embed_A = self.embeddings[ind_batch_A]
            embed_B = self.embeddings[ind_batch_B]
            return F.cosine_similarity(embed_A[:, None], embed_B[None], dim=-1).cpu().numpy()
        else:
            ind_batch_A = self._to_numpy(ind_batch_A).astype(int)
            ind_batch_B = self._to_numpy(ind_batch_B).astype(int)
            return self.similarity_matrix[np.ix_(ind_batch_A, ind_batch_B)]


class LPIPSMetric(Metric):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
        # use custom lpips version with batch support
        # https://github.com/david-klindt/PerceptualSimilarity/tree/batched
        from lpips import LPIPS
        self.loss_fn = LPIPS(net='alex').eval().to(self.device)
        self.num_scores = 6  # total and layers 1, 2, 3, 4, 5

    def preprocess(self, inputs):
        tensor = self._to_tensor(inputs)
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(torch.float32) / 255 * 2 - 1
        assert tensor.min() >= -1 and tensor.max() <= 1, "Input images must be normalized to [-1, 1]"        
        return tensor.to(torch.float32)
    
    def similarity(self, batch_A, batch_B) -> np.ndarray:
        """
        Compute LPIPS similarity between every pair in the two batches.
        Returns negative LPIPS distance as similarity.
        """
        with torch.no_grad():
            output = self.loss_fn(
                batch_A.to(self.device), batch_B.to(self.device), 
                normalize=True, retPerLayer=True)
            if self.ret_per_layer:
                output = torch.stack([output[0]] + output[1], dim=-1)
        return - output.detach().cpu().numpy()


class SSIMMetric(Metric):
    def __init__(self):
        super().__init__()
        from skimage.metrics import structural_similarity as ssim_func
        self.ssim_func = ssim_func

    def preprocess(self, inputs):
        inputs = self._to_numpy(inputs)
        if inputs.dtype == np.uint8:
            inputs = inputs.astype(np.float32) / 255 * 2 - 1
        assert inputs.min() >= -1 and inputs.max() <= 1, "Input images must be normalized to [-1, 1]"   
        return inputs.astype(np.float32)

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


class MSEMetric(Metric):
    def __init__(self):
        super().__init__()

    def preprocess(self, inputs):
        return self._to_tensor(inputs).to(torch.float32)

    def similarity(self, batch_A, batch_B) -> np.ndarray:
        """
        Compute negative mean squared error between every pair.
        """
        flat_A = torch.flatten(batch_A, start_dim=1)
        flat_B = torch.flatten(batch_B, start_dim=1)
        # Compute pairwise differences using broadcasting.
        mse = torch.mean((flat_A[:, None, :] - flat_B[None, :, :]) ** 2, dim=-1)
        return -mse
    

class CosineMetric(Metric):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def preprocess(self, inputs):
        return self._to_tensor(inputs).to(torch.float32)

    def similarity(self, batch_A, batch_B) -> np.ndarray:
        """
        Compute cosine similarity between every pair.
        """
        flat_A = torch.flatten(batch_A, start_dim=1)
        flat_B = torch.flatten(batch_B, start_dim=1)
        return F.cosine_similarity(flat_A[:, None], flat_B[None], dim=-1).cpu().numpy()


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
