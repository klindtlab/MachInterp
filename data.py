'''
To make a new dictionary:
- Each dictionary should be an identifier name. 
- Each key should be a seperate brain region.
- Values within a key are (in order):
  [0] path to image-containing folder OR path to an images.mat file
  [1] path to neural activations file
  [2] column name that neural activations are stored in
  [3] *optional future add on*: column name that images are stored in, if not imarray
  
Example usage:
import data
from data import loadData
identifier='Vinken2023'
region='IT'
images, activations=loadData(identifier,region)
images.shape, activations.shape

'''

# Import requirements and summarize available datasets
import scipy.io
import re
import torch
import os
import PIL
from PIL import Image
import numpy as np
import pickle

identifiers = {'BNNs': ['Allen2021', 'Cadena2019', 'Cadieu2024', 'Chang2021', 'Majaj2015', 'Vinken2023']}
base_dir = '/grid/klindt/data/Neuro'

# Datasets (alphabetical)
Allen2021 = {
    'None': [base_dir + '/Allen2021/nsd_data/nsd_data/stimulus_metadata.pkl',
           base_dir + '/Allen2021/nsd_data/nsd_data/neural_data.pkl',
           [1,2,5,7]] #subjects each have own column
}

Cadena2019 = {
    'V1': [base_dir + '/Cadena2019/data_binned_responses/cadena_ploscb_data.pkl',
           base_dir + '/Cadena2019/data_binned_responses/cadena_ploscb_data.pkl',
           [0,1,2,3]] #repetitions each have their own column   
}

Cadieu2024 = {
    'IT': [base_dir + '/Cadieu2024/IT/images-IT/',
           base_dir + '/Cadieu2024/IT/neural_responses-IT.mat',
           'neural_responses'],
    'V4': [base_dir + '/Cadieu2024/V4/images-V4/',
           base_dir + '/Cadieu2024/V4/neural_responses-V4.mat',
           'neural_responses']
}

Chang2021 = {
    'AM': [base_dir + '/Chang2021/AM/stimuli/',
           base_dir + '/Chang2021/AM/face_response.mat',
           'face_response'],
}

Majaj2015 = {
    'IT': [base_dir + '/Majaj2015/IT/images-IT/',
           base_dir + '/Majaj2015/IT/neural_responses-IT.mat',
           'neural_responses'],
    'V4': [base_dir + '/Majaj2015/V4/images-V4/',
           base_dir + '/Majaj2015/V4/neural_responses-V4.mat',
           'neural_responses']
}

Vinken2023 = {
    'IT': [base_dir + '/Vinken2023/IT/images.mat',
           base_dir + '/Vinken2023/IT/neural.mat',
           'R'],
}

# Load images
def loadImages(identifier, region):
    # Setup
    data_dict = globals().get(identifier)
    if data_dict is None:
        raise ValueError(f"Invalid identifier '{identifier}'.")
    if region not in data_dict:
        raise ValueError(f"Invalid region '{region}'.")
    data_log = data_dict.get(region)
    img_path = data_log[0]
    activations_path = data_log[1] 
    colname = data_log[2] 
    
    # Option 1: original path led to images.mat file
    if img_path.endswith('images.mat'): 
        inputs = scipy.io.loadmat(img_path)['imarray']
        channels_idx = [i for i, num in enumerate(inputs.shape) if num == 3][0]
        img_idx = [i for i, num in enumerate(inputs.shape) if num == np.max(inputs.shape)][0]
        h_idx, w_idx = [i for i, num in enumerate(inputs.shape) if num != np.max(inputs.shape) and num != 3]
        inputs = np.transpose(inputs, (img_idx, channels_idx, h_idx, w_idx))
        inputs = torch.from_numpy(inputs)
        
    # Option 2: load with metadata (Allen):
    elif img_path.endswith('stimulus_metadata.pkl'): #allen
        with open(base_dir + '/Allen2021/nsd_data/nsd_data/stimulus_metadata.pkl', "rb") as f:
            stimulus_metadata = pickle.load(f)
        image_files = []
        coco_categs, coco_supercategs = [], []
        for image_id in range(1000):
            image_files.append(
                os.path.join('/grid/klindt/data/Neuro/Allen2021/nsd_data/nsd_data/','stimulus_images',stimulus_metadata.image_name[image_id])
            )
            coco_categs.append(
                eval(stimulus_metadata.coco_categs[image_id])[0]
            )
            coco_supercategs.append(
                eval(stimulus_metadata.coco_supercategs[image_id])[0]
            )
        # extract indices for classes
        classes, labels = np.unique(coco_categs, return_inverse=True)
        classes_sup, labels_sup = np.unique(coco_supercategs, return_inverse=True)

        inputs = []
        for f in image_files:
            im = Image.open(f)
            im = im.resize((224, 224))
            inputs.append(np.transpose(np.array(im), (2, 0, 1)))
        inputs = np.array(inputs)
        #fix shape
        channels_idx = [i for i, num in enumerate(inputs.shape) if num == 3][0]
        img_idx = [i for i, num in enumerate(inputs.shape) if num == np.max(inputs.shape)][0]
        h_idx, w_idx = [i for i, num in enumerate(inputs.shape) if num != np.max(inputs.shape) and num!=3]
        inputs = np.transpose(inputs, (img_idx, channels_idx, h_idx, w_idx))
        inputs = torch.from_numpy(inputs)

    #Option 3: .pkl files, black and white handling (Cadena)
    elif img_path.endswith('cadena_ploscb_data.pkl'):  # Cadena
        with open(img_path, 'rb') as g:
            loaded_data = pickle.load(g)
        inputs = loaded_data['images']  # shape: [N, H, W]
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        inputs = inputs.unsqueeze(1)  # [N, 1, H, W]
        inputs = inputs.repeat(1, 3, 1, 1)  # [N, 3, H, W]
        

    # Option 4: you created images.mat from images without updating data.py's dictionary
    elif 'images.mat' in os.listdir(os.path.join(base_dir, identifier, region)):
        inputs = scipy.io.loadmat(os.path.join(base_dir, identifier, region, 'images.mat'))['imarray']
        channels_idx = [i for i, num in enumerate(inputs.shape) if num == 3][0]
        img_idx = [i for i, num in enumerate(inputs.shape) if num == np.max(inputs.shape)][0]
        h_idx, w_idx = [i for i, num in enumerate(inputs.shape) if num != np.max(inputs.shape) and num != 3]
        inputs = np.transpose(inputs, (img_idx, channels_idx, h_idx, w_idx))
        inputs = torch.from_numpy(inputs)

    # Option 5: create images.mat and save to appropriate path for faster loading
    else:
        img_files = [f for f in os.listdir(img_path) if f.endswith('.png') or f.endswith('.tiff') or f.endswith('.jpg')]
        img_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        images = [Image.open(os.path.join(img_path, file)).convert('RGB') for file in img_files]
        img_arrays = [np.array(image) for image in images]
        inputs = np.stack(img_arrays, axis=0)
        channels_idx = [i for i, num in enumerate(inputs.shape) if num == 3][0]
        img_idx = [i for i, num in enumerate(inputs.shape) if num == np.max(inputs.shape)][0]
        h_idx, w_idx = [i for i, num in enumerate(inputs.shape) if num != np.max(inputs.shape) and num!=3]
        inputs = np.transpose(inputs, (img_idx, channels_idx, h_idx, w_idx))
        save_dir = os.path.join(base_dir, identifier, region, 'images.mat')
        scipy.io.savemat(save_dir, {'imarray':inputs})
        print(f'images.mat saved to {save_dir}')
    num_imgs = inputs.shape[0]  
    
    return inputs,num_imgs
  
# Load neural activations    
def loadActivations(identifier, region, num_imgs, subject_idx=0):
    # Setup
    data_dict = globals().get(identifier, None)
    if data_dict is None:
        raise ValueError(f"Invalid identifier '{identifier}'.")
    data_log = data_dict.get(region, None)
    if data_log is None:
        raise ValueError(f"Invalid region '{region}'.")
    img_path = data_log[0]
    activations_path = data_log[1] 
    colname = data_log[2] 
    
    # Option 1: load .mat files
    if activations_path.endswith('.mat'):
        activations = scipy.io.loadmat(activations_path)[colname]
        activations = torch.tensor(activations, dtype = torch.float32)
    
    # Option 2: load .pt files
    elif activations_path.endswith('.pt'):
        activations = torch.load(activations_path)[colname]
    
    # Option 3: load .pkl files with metadata (Allen)
    elif activations_path.endswith('neural_data.pkl'): #allen
        #load activations
        with open(os.path.join(base_dir,'Allen2021/nsd_data/nsd_data/neural_data.pkl'), "rb") as f:
            activations = pickle.load(f)
        #select one subject's activations for area V1d
        subject=Allen2021['None'][2][subject_idx]
        activations = np.array(activations[subject]['V1d'])
        activations = torch.tensor(activations, dtype = torch.float32)
   
    # Option 4: load .pkl files by column, with 1 repetition (Cadena)
    elif activations_path.endswith('cadena_ploscb_data.pkl'): #cadena
        with open('/grid/klindt/data/Neuro/Cadena2019/data_binned_responses/cadena_ploscb_data.pkl', 'rb') as g:
            loaded_data = pickle.load(g)
        activations = loaded_data['responses'][0]
        activations = torch.tensor(activations, dtype = torch.float32)
        activations = torch.nan_to_num(activations, nan=1e-6)  # Replace NaNs with 0.0
        
    else:
        raise ValueError(f"Unsupported file type: {activations_path}")

    if activations.shape[0] != num_imgs:
        activations = activations.T

    return activations


def loadData(identifier, region, subject_idx=None): 

    images, num_imgs = loadImages(identifier, region)
    activations = loadActivations(identifier, region, num_imgs)
    
    return images, activations #unindent after allen is resolved