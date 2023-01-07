import torch 
from torch import nn 
import os
from models import ResNet
from dataclasses import dataclass
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class DataStorage:
    path_to_save: str = './data/'
    path_to_load: str = "https://www.kaggle.com/competitions/plant-seedlings-classification/data"

class PreprocessConfiguration:
    batch_size: int = 32
    resize:int = 224
    train_size: float = 0.8
    image_url_for_std: str = './plant-seedlings-classification/train/*/*.*'
    image_url_for_train: str = './plant-seedlings-classification/train/'
    num_workers:int = os.cpu_count()
    prediction_data:bool = False

class TrainingConfiguration:
    model_name: str = 'resnet_100_epochs'
    epochs: int=150
    learning_rate: float = 0.001
    loss_criteron :nn = nn.CrossEntropyLoss()
    model: nn.Module = ResNet().to(device)
    optimizer: torch.optim = torch.optim.Adam
    
