import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import cv2


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
        Accepts a list of images or a numpy array.
        """
        # If list of file paths, load them first.
        if isinstance(inputs, list):
            processed = [self.model_preprocess(img) for img in inputs]
            return np.stack(processed)
        elif isinstance(inputs, np.ndarray):
            return np.stack([self.model_preprocess(img) for img in inputs])
        elif torch.is_tensor(inputs):
            return inputs  # Assume already preprocessed.
        else:
            return inputs

    def similarity(self, batch_A, batch_B) -> np.ndarray:
        """
        Compute cosine similarity between embeddings.
        Converts numpy arrays to torch tensors if needed.
        """
        if isinstance(batch_A, np.ndarray):
            batch_A = torch.from_numpy(batch_A).to(self.device)
        if isinstance(batch_B, np.ndarray):
            batch_B = torch.from_numpy(batch_B).to(self.device)
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
        self.to_tensor = ToTensor()

    def preprocess(self, inputs):
        """
        Preprocess images for LPIPS: convert to tensor, scale to [-1, 1].
        Accepts a list of images or a numpy array.
        """
        if isinstance(inputs, list):
            processed = [(2 * self.to_tensor(img) - 1) for img in inputs]
            return torch.stack(processed).to(self.device)
        elif isinstance(inputs, np.ndarray):
            processed = [(2 * self.to_tensor(img) - 1) for img in inputs]
            return torch.stack(processed).to(self.device)
        elif torch.is_tensor(inputs):
            return inputs
        else:
            return inputs

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


class MSEMetric(Metric):
    def __init__(self, device: str = 'cpu'):
        super().__init__(device)

    def preprocess(self, inputs):
        """
        For MSE, no heavy preprocessing is needed.
        Accepts numpy arrays, torch tensors, or a list of file paths.
        """
        if isinstance(inputs, list):
            processed = []
            for item in inputs:
                if isinstance(item, str):
                    # Load image from file and convert BGR to RGB.
                    img = cv2.imread(item)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    processed.append(img)
                else:
                    processed.append(item)
            return np.array(processed)
        elif isinstance(inputs, np.ndarray):
            return inputs
        elif torch.is_tensor(inputs):
            return inputs.cpu().numpy()
        else:
            return inputs

    def similarity(self, batch_A, batch_B) -> np.ndarray:
        """
        Compute negative mean squared error between every pair.
        """
        # Flatten each image.
        A = np.reshape(batch_A, (batch_A.shape[0], -1))
        B = np.reshape(batch_B, (batch_B.shape[0], -1))
        # Compute pairwise differences using broadcasting.
        mse = np.mean((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return -mse


class SSIMMetric(Metric):
    def __init__(self, device: str = 'cpu'):
        super().__init__(device)
        from skimage.metrics import structural_similarity as ssim_func
        self.ssim_func = ssim_func

    def preprocess(self, inputs):
        """
        Preprocess images for SSIM.
        Accepts numpy arrays, torch tensors, or a list of file paths.
        """
        if isinstance(inputs, list):
            processed = []
            for item in inputs:
                if isinstance(item, str):
                    img = cv2.imread(item)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    processed.append(img)
                else:
                    processed.append(item)
            return np.array(processed)
        elif isinstance(inputs, np.ndarray):
            return inputs
        elif torch.is_tensor(inputs):
            return inputs.cpu().numpy()
        else:
            return inputs

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
    

class CosineMetric(Metric):
    def __init__(self, device: str = 'cpu'):
        super().__init__(device)

    def preprocess(self, inputs):
        if isinstance(inputs, list):
            return np.array(inputs)
        elif isinstance(inputs, np.ndarray):
            return inputs
        elif torch.is_tensor(inputs):
            return inputs.cpu().numpy()
        else:
            return inputs

    def similarity(self, batch_A, batch_B) -> np.ndarray:
        A = np.reshape(batch_A, (batch_A.shape[0], -1))
        B = np.reshape(batch_B, (batch_B.shape[0], -1))
        epsilon = 1e-8
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + epsilon)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + epsilon)
        return np.dot(A_norm, B_norm.T)


class LabelMetric(Metric):
    def __init__(self, device: str = 'cpu'):
        super().__init__(device)

    def preprocess(self, inputs):
        """
        For label-based metrics, assume inputs are already labels (ints, strings, etc.)
        """
        return inputs

    def similarity(self, batch_A, batch_B) -> np.ndarray:
        """
        Compute similarity based on equality: 1 if labels are equal, 0 otherwise.
        """
        batch_A = np.array(batch_A)
        batch_B = np.array(batch_B)
        return (batch_A[:, None] == batch_B[None, :]).astype(int)