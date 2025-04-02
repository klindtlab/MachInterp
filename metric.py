import numpy as np
import torch
import torch.nn.functional as F


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


class DreamSimMetric(Metric):
    def __init__(self, device: str = 'cpu'):
        super().__init__(device)
        from dreamsim import dreamsim
        self.model, self.model_preprocess = dreamsim(pretrained=True, device=device)
        self.embed_fn = self.model.embed

    def preprocess(self, inputs):
        """
        Preprocess inputs using the DreamSim model's preprocessing.
        Accepts numpy arrays or torch tensors.
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs, dtype=torch.float32, device=self.device)
        elif torch.is_tensor(inputs):
            inputs = inputs.to(torch.float32, device=self.device)
        else:
            raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")
        return self.model_preprocess(inputs)

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


class LPIPSMetric(Metric):
    def __init__(self, device: str = 'cpu', ret_per_layer: bool = False):
        super().__init__(device)
        # use custom lpips version with batch support
        # https://github.com/david-klindt/PerceptualSimilarity/tree/batched
        from lpips import LPIPS
        self.loss_fn = LPIPS(net='alex').to(self.device)
        self.ret_per_layer = ret_per_layer

    def preprocess(self, inputs):
        """
        Preprocess images for LPIPS: convert to tensor, scale to [-1, 1].
        Accepts numpy arrays or torch tensors.
        """
        assert inputs.min() >= -1 and inputs.max() <= 1, "Input images must be normalized to [-1, 1]"
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs, dtype=torch.float32, device=self.device)
        elif torch.is_tensor(inputs):
            inputs = inputs.to(torch.float32, device=self.device)
        else:
            raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")

    def similarity(self, batch_A, batch_B) -> np.ndarray:
        """
        Compute LPIPS similarity between every pair in the two batches.
        Returns negative LPIPS distance as similarity.
        """
        with torch.no_grad():
            _, output = self.loss_fn(batch_A, batch_B, normalize=True, retPerLayer=self.ret_per_layer)
            if self.ret_per_layer:
                output = torch.stack(output, dim=-1)
        return - output.detach().cpu().numpy()


class SSIMMetric(Metric):
    def __init__(self, device: str = 'cpu'):
        super().__init__(device)
        from skimage.metrics import structural_similarity as ssim_func
        self.ssim_func = ssim_func

    def preprocess(self, inputs):
        """
        Preprocess images for SSIM.
        Accepts numpy arrays or torch tensors.
        """
        if isinstance(inputs, np.ndarray):
            return inputs
        elif torch.is_tensor(inputs):
            return inputs.cpu().numpy()
        else:
            raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")

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
                img_A = batch_A[i]
                img_B = batch_B[j]
                # Convert RGB to grayscale if needed.
                if img_A.ndim == 3 and img_A.shape[2] == 3:
                    img_A_gray = np.dot(img_A[...,:3], [0.2989, 0.5870, 0.1140])
                    img_B_gray = np.dot(img_B[...,:3], [0.2989, 0.5870, 0.1140])
                else:
                    img_A_gray = img_A
                    img_B_gray = img_B
                similarity_matrix[i, j] = self.ssim_func(img_A_gray, img_B_gray)
        return similarity_matrix
    

def flatten_images(images):
    """
    Flatten images to 2D arrays.
    """
    return np.reshape(images, (images.shape[0], -1))


class MSEMetric(Metric):
    def __init__(self, device: str = 'cpu'):
        super().__init__(device)

    def preprocess(self, inputs):
        """
        For MSE, no heavy preprocessing is needed.
        Accepts numpy arrays or torch tensors.
        """
        if isinstance(inputs, np.ndarray):
            return inputs
        elif torch.is_tensor(inputs):
            return inputs.cpu().numpy()
        else:
            raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")

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
    def __init__(self, device: str = 'cpu', epsilon: float = 1e-8):
        super().__init__(device)
        self.epsilon = epsilon

    def preprocess(self, inputs):
        if isinstance(inputs, np.ndarray):
            return inputs
        elif torch.is_tensor(inputs):
            return inputs.cpu().numpy()
        else:
            raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")

    def similarity(self, batch_A, batch_B) -> np.ndarray:
        flat_A = flatten_images(batch_A)
        flat_B = flatten_images(batch_B)
        A_norm = flat_A / (np.linalg.norm(flat_A, axis=1, keepdims=True) + self.epsilon)
        B_norm = flat_B / (np.linalg.norm(flat_B, axis=1, keepdims=True) + self.epsilon)
        return np.dot(A_norm, B_norm.T)


class LabelMetric(Metric):
    def __init__(self, device: str = 'cpu'):
        super().__init__(device)

    def preprocess(self, inputs):
        """
        For label-based metrics, assume inputs are already labels.
        Accepts numpy arrays or torch tensors.
        """
        if isinstance(inputs, np.ndarray):
            return inputs
        elif torch.is_tensor(inputs):
            return inputs.cpu().numpy()
        else:
            raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")

    def similarity(self, batch_A, batch_B) -> np.ndarray:
        """
        Compute similarity based on equality: 1 if labels are equal, 0 otherwise.
        """
        return (np.array(batch_A)[:, None] == np.array(batch_B)[None, :]).astype(float)