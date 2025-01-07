import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import ToTensor


def get_lpips(device):
    """
    Get LPIPS similarity metric and preprocessing functions.

    Args:
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        tuple: (sim_metric, preprocess_ds)
            - sim_metric: Function that computes LPIPS similarity between two image tensors
            - preprocess_ds: Function that preprocesses a dataset of images for LPIPS
    """
    from lpips import LPIPS
    loss_fn = LPIPS(net='alex').to(device)
    to_tensor_inst = ToTensor()

    def sim_metric(im_tensor0, im_tensor1):
        assert len(im_tensor0.shape) in (3, 4)
        assert len(im_tensor1.shape) in (3, 4)
        if len(im_tensor0.shape) == 3:
            im_tensor0 = im_tensor0.unsqueeze(0)
        if len(im_tensor1.shape) == 3:
            im_tensor1 = im_tensor1.unsqueeze(0)

        loss_fn.eval()
        with torch.no_grad():
            output = - loss_fn(im_tensor0.to(device), im_tensor1.to(device))

        return output

    def preprocess(im):
        return (2 * to_tensor_inst(im) - 1).unsqueeze(0)
    
    def collator(batch: list):
        output = [preprocess(image) for image in batch]
        return torch.cat(output, dim=0)

    def preprocess_ds(ds):
        ds_loader = DataLoader(ds, batch_size=64,
                               shuffle=False, num_workers=2*torch.cuda.device_count(),
                               collate_fn=collator)

        try:
            from tqdm import tqdm
            loader_loop = tqdm(enumerate(ds_loader), total=len(ds_loader) )
        except ImportError:
            loader_loop = enumerate(ds_loader)

        im_tensor_set=[]
        print("Preprocessing images for LPIPS...")
        for _ , X in loader_loop:
            im_tensor_set.append( X.to(device) )

        return torch.cat(im_tensor_set, dim=0)


    return sim_metric , preprocess_ds


def get_dreamsim(device):
    """
    Get DreamSim similarity metric and preprocessing functions.

    Args:
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        tuple: (sim_metric, preprocess_embed_ds)
            - sim_metric: Function that computes cosine similarity between DreamSim embeddings
            - preprocess_embed_ds: Function that preprocesses and embeds a dataset of images
    """
    from dreamsim import dreamsim

    model, preprocess = dreamsim(pretrained=True, device=device)

    embed_fn = model.embed

    def sim_metric(embed1, embed2):
        if len(embed1.shape)==1:
            embed1 = embed1.unsqueeze(0)
        if len(embed2.shape)==1:
            embed2 = embed2.unsqueeze(0)

        return F.cosine_similarity(embed1[:, None], embed2[None], dim=-1)
    
    def collator(batch: list):
        output = [preprocess(image) for image in batch]
        return torch.cat(output, dim=0)

    def preprocess_embed_ds(ds):
        ds_loader = DataLoader(ds, batch_size=64,
                               shuffle=False, num_workers=2*torch.cuda.device_count(),
                               collate_fn=collator)

        try:
            from tqdm import tqdm
            loader_loop = tqdm(enumerate(ds_loader), total=len(ds_loader) )
        except ImportError:
            loader_loop = enumerate(ds_loader)

        print("Preprocessing images for DREAMSIM...")
        embedding=[]
        model.eval()
        with torch.no_grad():
            for _ , X in loader_loop:
                embedding.append( embed_fn(X.to(device)) )

        return torch.cat(embedding, dim=0)

    return sim_metric, preprocess_embed_ds


def get_metric_preprocess(metric_type: str, device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')):
    """
    Get similarity metric and preprocessing functions for specified metric type.

    Args:
        metric_type: Type of metric to get ('dreamsim' or 'lpips')
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        tuple: (metric_fn, preprocess_fn)
            - metric_fn: Function that computes similarity between images/embeddings
            - preprocess_fn: Function that preprocesses images for the metric

    Raises:
        AssertionError: If metric_type is not 'dreamsim' or 'lpips'
    """
    metric_type = metric_type.lower()
    assert metric_type in ["dreamsim", "lpips"]

    if metric_type == "dreamsim":
        return get_dreamsim(device)
    if metric_type == "lpips":
        return get_lpips(device)
    

class Metric_Preprocess:
    """
    A singleton class that stores and manages metric and preprocessing functions.

    This class caches metric functions and their corresponding preprocessing functions
    to avoid reloading them multiple times. It uses a tuple of (metric_type, device) 
    as keys to store and retrieve the functions.

    Attributes:
        _instance: Class variable storing the singleton instance
        _metric: Dictionary storing metric functions keyed by (metric_type, device)
        _preprocess: Dictionary storing preprocessing functions keyed by (metric_type, device)

    Methods:
        __call__(metric_type, device): Get both metric and preprocess functions
        process(dataset, metric_type, device): Preprocess a dataset using cached function
        get_metric(metric_type, device): Get just the metric function
    """
    _instance = None
    _metric = {}
    _preprocess = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if one doesn't already exist
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, metric_type: str=None, device: str=None):
        """
        Initialize or update the singleton instance.

        Args:
            metric_type: Type of metric to load ('dreamsim' or 'lpips')
            device: Device to load the metric on ('cpu' or 'cuda')
        """
        # Update the value every time the instance is called
        if metric_type is not None and device is not None:
            key = (metric_type, device)
            if not key in self._metric.keys():
                self._metric[key] , self._preprocess[key] = get_metric_preprocess(metric_type , device)

    def __call__(self, metric_type: str, device: str):
        """
        Get metric and preprocessing functions for given type and device.

        Args:
            metric_type: Type of metric ('dreamsim' or 'lpips')
            device: Device to use ('cpu' or 'cuda')

        Returns:
            Tuple of (metric_function, preprocess_function)
        """
        key = (metric_type, device)
        if not key in self._metric.keys():
            self._metric[key] , self._preprocess[key] = get_metric_preprocess(metric_type , device)
        return self._metric[key] , self._preprocess[key]
    
    def process(self, dataset, metric_type: str, device: str):
        """
        Preprocess a dataset using cached preprocessing function.

        Args:
            dataset: Dataset to preprocess
            metric_type: Type of metric preprocessing to use
            device: Device to preprocess on

        Returns:
            Preprocessed dataset
        """
        key = (metric_type, device)
        if not key in self._metric.keys():
            self._metric[key] , self._preprocess[key] = get_metric_preprocess(metric_type , device)
        
        return self._preprocess[key](dataset)
    
    def get_metric(self, metric_type: str, device: str):
        """
        Get metric function for given type and device.

        Args:
            metric_type: Type of metric to get
            device: Device to get metric for

        Returns:
            Metric function
        """
        key = (metric_type, device)
        if not key in self._metric.keys():
            self._metric[key] , self._preprocess[key] = get_metric_preprocess(metric_type , device)
        return self._metric[key]
    

def process(dataset, metric_type: str, device):
    """
    Preprocess a dataset using the specified metric type and device.

    Args:
        dataset: Dataset to preprocess
        metric_type: Type of metric preprocessing to use ('dreamsim' or 'lpips')
        device: Device to preprocess on ('cpu' or 'cuda')

    Returns:
        Preprocessed dataset
    """
    wrapper = Metric_Preprocess()
    return wrapper.process(dataset, metric_type, device)


def get_metric(metric_type: str, device):
    """
    Get metric function for the specified type and device.

    Args:
        metric_type: Type of metric to get ('dreamsim' or 'lpips')
        device: Device to get metric for ('cpu' or 'cuda')

    Returns:
        Metric function for computing similarity between images
    """
    wrapper = Metric_Preprocess()
    return wrapper.get_metric(metric_type, device)
