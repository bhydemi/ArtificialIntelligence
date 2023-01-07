"""
This scripts helps to download from open source data and 
preprocess data and make them ready to be used in the deep learning models
"""


import numpy as np
import cv2 
import tqdm 
import random
import glob 
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize, TrivialAugmentWide, Normalize, AugMix, AutoAugment, RandAugment
from torch.utils.data import DataLoader, Dataset
import opendatasets as od
from configuration import PreprocessConfiguration, DataStorage
from copy import copy

def download_kaggle_data(Data=DataStorage):
    """
    Downloads Data from Opensource Sites Like Kaggle
    """
    od.download(DataStorage.path_to_load)

def mean_std_images(image_url:str, sample:int) -> tuple:
    """
    Takes in a random sample of images and returns the mean and 
    standard deviation of the all channels regarding all images
    """
    means = np.array([0, 0, 0], dtype=np.float32)
    stds = np.array([0, 0, 0], dtype=np.float32)
    total_images = 0
    randomly_sample = sample
    for f in tqdm.tqdm(random.sample(glob.glob(image_url, recursive = True), randomly_sample)):
        img = cv2.imread(f)
        means += img.mean(axis=(0,1))
        stds += img.std(axis=(0,1))
        total_images += 1
    means = means / (total_images * 255.)
    stds = stds / (total_images * 255.)
    return means, stds


def preprocess_image_folder_data( preprocessing_configuration = PreprocessConfiguration()):
    """
    Preprocessing Downloaded Data and Returns DataLoader Files for Training And Test Data
    Mean & Standard Deviation of Sample Images and Classes of Data
    """
    download_kaggle_data()
    print('Step 1: Preprocessing Image')
    all_files =  glob.glob(preprocessing_configuration.image_url_for_train)

    print('Step 1.1: Randomly calculating mean and standard for Train Transform normalize')

    mean, std = mean_std_images(preprocessing_configuration.image_url_for_std, 3000) 
   

    print('Step 1.2: Loading Image from folders')
    
    full_train_dataset = ImageFolder(
        root=preprocessing_configuration.image_url_for_train,
        transform= None
            )
    if not preprocessing_configuration.prediction_data:
        train_size = int(preprocessing_configuration.train_size * len(full_train_dataset))
        test_size = len(full_train_dataset) - train_size
    
        print('Step 1.3: Train/Test Split Datasets')
    
        train_data, test_data = torch.utils.data.random_split(full_train_dataset, [train_size, test_size])
        train_data.dataset = copy(full_train_dataset)
        train_data.dataset.transform = Compose([Resize((preprocessing_configuration.resize,preprocessing_configuration.resize)), 
                        RandAugment(),
                        ToTensor(),
                        Normalize(mean=mean,std=std)])
        test_data.dataset.transform = Compose([Resize((preprocessing_configuration.resize,preprocessing_configuration.resize)), 
                        ToTensor(),
                        Normalize(mean=mean,std=std)])

    if preprocessing_configuration.prediction_data:
        valid_loader = DataLoader(
        full_train_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=preprocessing_configuration.num_workers, pin_memory=True, 
        )
        return valid_loader


    BATCH_SIZE = preprocessing_configuration.batch_size

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=preprocessing_configuration.num_workers, pin_memory=True
    )
    
    valid_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=preprocessing_configuration.num_workers, pin_memory=True, 
    )


    return mean, std, train_loader, valid_loader, full_train_dataset.classes
