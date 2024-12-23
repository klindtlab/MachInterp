import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import ToTensor


def get_lpips(device):
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
    metric_type = metric_type.lower()
    assert metric_type in ["dreamsim", "lpips"]

    if metric_type == "dreamsim":
        return get_dreamsim(device)
    if metric_type == "lpips":
        return get_lpips(device)
    

class Metric_Preprocess:
    """
    A simple mutable singleton class.
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
        # Update the value every time the instance is called
        if metric_type is not None and device is not None:
            key = (metric_type, device)
            if not key in self._metric.keys():
                self._metric[key] , self._preprocess[key] = get_metric_preprocess(metric_type , device)


    def __call__(self, metric_type: str, device: str):
        key = (metric_type, device)
        if not key in self._metric.keys():
            self._metric[key] , self._preprocess[key] = get_metric_preprocess(metric_type , device)
        return self._metric[key] , self._preprocess[key]
    
    def process(self, dataset, metric_type: str, device: str):
        key = (metric_type, device)
        if not key in self._metric.keys():
            self._metric[key] , self._preprocess[key] = get_metric(metric_type , device)
        
        return self._preprocess[key](dataset)
    
    def get_metric(self, metric_type: str, device: str):
        key = (metric_type, device)
        if not key in self._metric.keys():
            self._metric[key] , self._preprocess[key] = get_metric(metric_type , device)
        return self._preprocess[key]
    

def process(dataset, metric_type: str, device):
    wrapper = Metric_Preprocess()
    return wrapper.process(dataset, metric_type, device)


def get_metric(metric_type: str, device):
    wrapper = Metric_Preprocess()
    return wrapper.get_metric(metric_type, device)
