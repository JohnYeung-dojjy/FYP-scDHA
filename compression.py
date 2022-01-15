import argparse
import pandas as pd
import numpy as np

import torch
from torchvision import transforms
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import torch.optim as optim                          # optimization

import os
from tqdm import tqdm                                # for progress bar
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from numba import njit

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the gpu")
else:
    device = torch.device("cpu")
    print("running on the cpu")
    
import denoise, encode, paper_encode

from denoise import non_negative_kernel_autoencoder
from encode import stacked_bayesian_autoencoder
from paper_encode import paper_encoder


def normalization(data):
    # non-negative normalization, all data are from 0-1
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)

    # avoid 0 division
    max_minus_min = (data_max - data_min)
    zero_loc = np.where(max_minus_min == 0)
    max_minus_min[zero_loc] = 1
    
    normalized_data = (data - data_min) / max_minus_min
    
    return normalized_data


def pipeline(path_name, denoise_batch_size, denoise_epochs, denoise_plot, encode_batch_size, encode_epochs, vae_choice, retrain=False):

    if os.path.isfile(f'latent/{path_name}.npy') and not retrain:
        return np.load(f'latent/{path_name}.npy')
    else:
        if os.path.isfile(f'npy_data/{path_name}.npy'):
            # read the npy file
            data = np.load(f'npy_data/{path_name}.npy')
        else:
            # read the data to panda_df from path
            data_df = pd.read_csv(f"data/{path_name}.csv")
            # make the data in the form of np
            data = np.array(data_df)
            # save the nparray
            np.save(f'npy_data/{path_name}.npy', data)
        
        # normalize the data
        # normalized_data = normalization(data)
        normalizer = preprocessing.MinMaxScaler()
        normalized_data = normalizer.fit_transform(data)
        

        # define the non_negative_kerel_autoencoder object
        XNeg_kernel_autoencoder = non_negative_kernel_autoencoder(normalized_data.shape[1])
        print(XNeg_kernel_autoencoder)

        # Train the model
        denoise.train_model(XNeg_kernel_autoencoder, normalized_data, EPOCHS=denoise_epochs, BATCH_SIZE=denoise_batch_size)
        
        # stop updating the model's parameters
        XNeg_kernel_autoencoder.eval()
        
        # choose 5000 most important features
        denoised_data = XNeg_kernel_autoencoder.K_MostImportant_features(normalized_data, 5000, plot=denoise_plot)
        # print(denoised_data.max())
        # denoised_data = normalization(denoised_data)
        
        latent = []
        
        if vae_choice == 'mine':
            # define the stacked_bayesian_autoencoder object
            VAE = stacked_bayesian_autoencoder(original_dim=denoised_data.size()[1], im_dim=64, lat_dim=15)
            print(VAE)
            # Train the model
            encode.train_model(VAE, denoised_data, beta = 50, EPOCHS_0=encode_epochs[0], EPOCHS_1=encode_epochs[1], BATCH_SIZE=encode_batch_size)
        elif vae_choice == "paper":
            # define the stacked_bayesian_autoencoder object
            VAE = paper_encoder(original_dim=denoised_data.size()[1], im_dim=64, lat_dim=15)
            print(VAE)
            # Train the model
            paper_encode.train_model(VAE, denoised_data, beta = 50, EPOCHS_0=encode_epochs[0], EPOCHS_1=encode_epochs[1], BATCH_SIZE=encode_batch_size)
        
        # stop updating the model's parameters
        VAE.eval()
        
        # return the latent variable
        with torch.no_grad():
            latent.append(VAE.encode_mu(denoised_data).cpu().numpy())
            # print(latent.requires_grad)
            # torch.save(latent, f'latent/{path_name}.pt')
            # print(latent.cpu().numpy())
            # np.save(f'latent/{path_name}.npy', latent.cpu().numpy())
        return latent

if __name__ == '__main__':
    pipeline()
