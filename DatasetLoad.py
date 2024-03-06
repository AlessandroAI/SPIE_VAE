import torchgeo.datasets as geo
import os
import tempfile
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchgeo.datasets import NAIP, ChesapeakeDE, stack_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler
from torch.utils.data import random_split
from torch import random
from torchvision import transforms as T
import random 
import numpy as np
from torch.utils.data import Subset

def transform_fn(data):
    transform = T.ToTensor()
    data['image'] = transform(data['image'])
    return data

tranform = transform_fn

def dataset(dataset_class, dataset_path, split_type, shuffle):
    # Define the dataset variable outside the if statement
    dataset = None

    if dataset_class == 'EuroSAT':
        
        ## The bands are the 13 bands of Sentinel-2 pull from torch geo library teh datset and split type
        
        dataset = geo.EuroSAT(root=dataset_path, split= split_type, bands=('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B08A', 'B09', 'B10', 'B11', 'B12')
                              , download=True, checksum=False)
        
        if shuffle:
            
            # Create a list of indices
            indices = list(range(len(dataset)))

            # Shuffle the indices
            np.random.shuffle(indices)

            # Create a new dataset in the shuffled order
            dataset = Subset(dataset, indices)

   
    return dataset


def dataset_RGB(dataset_class, dataset_path, split_type):
    # Define the dataset variable outside the if statement
    dataset = None

    if dataset_class == 'EuroSAT':
        
        ## The bands are the RGB bands of Sentinel-2 pull from torch geo library teh datset and split type 

        dataset = geo.EuroSAT(root=dataset_path, split= split_type, bands=( 'B02', 'B03', 'B04') , download= True, checksum=False)

  
def get_classes(dataset):
    return dataset.classes

def get_num_classes(dataset):
    return dataset.num_classes
