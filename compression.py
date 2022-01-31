import argparse
import pandas as pd
import numpy as np

import torch
from torchvision import transforms
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch.utils.data import DataLoader

import os
from tqdm import tqdm                                # for progress bar
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# from numba import njit

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
import denoise, encode, paper_encode

from denoise import non_negative_kernel_autoencoder, K_MostImportant_features
from encode import stacked_bayesian_autoencoder
from paper_encode import paper_encoder


def normalization(data):
    # non-negative normalization, all data are from 0-1
    # normalize per sample -> use row min/max (axis=1)
    data_min = data.min(axis=1)
    data_max = data.max(axis=1)

    # avoid 0 division
    max_minus_min = (data_max - data_min)
    # zero_loc = np.where(max_minus_min == 0)
    # max_minus_min[zero_loc] = 1
    
    normalized_data = ((data.T - data_min) / max_minus_min)
    normalized_data[np.isnan(normalized_data)] = 0
    return normalized_data.T


def pipeline(path_name, is_plot_denoise, vae_choice, retrain=False):
    """
    load the data and compress it using the scDHA pipeline
    return the compressed latent data
    """
    # read file, if we have a generated latent, return it, else train it
    if os.path.isfile(f'latent/{path_name}.npy') and not retrain:
        data = np.load(f'latent/{path_name}.npy')
        print("loaded from latent")
        return data
    else:
        if os.path.isfile(f'npy_data/{path_name}.npy'):
            # read the npy file
            data = np.load(f'npy_data/{path_name}.npy')
        else:
            # read the data to panda_df from path
            data_df = pd.read_csv(f"data/{path_name}.csv")
            # make the data in the form of np
            data = np.array(data_df)
            # if range of data > 100, perform log transforms to data
            
            # save the nparray
            np.save(f'npy_data/{path_name}.npy', data)
        # print(data, data.max(), data.min())
        
        ######### training variables ###########
        wdecay = [1e-6, 1e-3]
        batch_size = max(round(len(data)/50), 2)
        denoise_epochs = 10
        encode_epochs = [10, 20]
        orginal_dim = data.shape[1]
        ########################################
        
        # different cell have different number of samples
        # lead to high expression if simply more samples
        # we can normalize/log the value in cell to avoid dominance
        # in scDHA they chose log2
        if data.max() - data.min() > 100:
            print("log2 is applied")
            data = np.log2(data + 1) # add 1 to pervent log(0)
            
        # normalize the data
        normalized_data = normalization(data)
        print(f"first column of normalized_data\n{normalized_data[:,0]}")
        
        device_data = torch.tensor(normalized_data, dtype=torch.float).to(device)

        # Train the model
        # denoise.train_model(XNeg_kernel_autoencoder, dataloader, EPOCHS=denoise_epochs, BATCH_SIZE=batch_size)
        
        print("training non-negative kernel autoencoder")
        # gene filtering
        Wsds = [] # standard deviations of Weights
        for _ in range(3): # repeat for 3 times and take the average variance of weights
            dataloader = DataLoader(device_data, batch_size=batch_size, shuffle=True, drop_last=True)
        
            # define the non_negative_kerel_autoencoder object
            XNeg_kernel_autoencoder = non_negative_kernel_autoencoder(orginal_dim, 32)
            XNeg_kernel_autoencoder.train()
            
            loss_function = nn.MSELoss()
            optimizer = torch.optim.AdamW(XNeg_kernel_autoencoder.parameters(), 
                                          lr=1e-3, eps=1e-7, weight_decay=wdecay[0])
            for epoch in range(denoise_epochs):
                optimizer.zero_grad()
                for data in dataloader:
                    # Forward
                    decoded_data = XNeg_kernel_autoencoder(data)

                    # Backward
                    loss = loss_function(decoded_data, data)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    
                    with torch.no_grad():
                        XNeg_kernel_autoencoder.encoder.weight.clamp_(min = 0) # inplace clamp
                    
                    # model.encoder.weight = Parameter(model.encoder.weight.detach().clamp(min = 0))
                    
                # Show progress
                if epoch%10 == 9:
                    print(f"epoch {epoch} Loss: ", loss.item())
        
            # calculate variance of each features
            # https://pytorch.org/docs/stable/generated/torch.var.html
            # https://pytorch.org/docs/stable/generated/torch.transpose.html
            # weight_var = torch.std(torch.transpose(self.encoder.weight, 0, 1), dim=1)
            Wsd = torch.std(XNeg_kernel_autoencoder.encoder.weight, 0, 1, dim=0)
            # print(weight_var.data)
            # rescale weight_var to [0, 1] for ploting use
            Wsd[torch.isnan(Wsd)] = 0 # I don't know why this is here
            Wsd = (Wsd - Wsd.min()) / (Wsd.max() - Wsd.min())
            Wsd[torch.isnan(Wsd)] = 0 # in case of 0 division
            
            Wsds.append(Wsd.cpu().detach()) # add to Wsds
        
        Wsds = torch.cat((Wsds[0], Wsds[1], Wsds[2]), 0).mean(dim=0) # mean of each row
        
        # stop updating the model's parameters
        XNeg_kernel_autoencoder.eval()
        
        # choose 5000 most important features
        denoised_data = denoise.K_MostImportant_features(normalized_data, Wsds, plot=is_plot_denoise)
        # print(denoised_data.max())
        # denoised_data = normalization(denoised_data)
        
        # latent generating
        latent = []
        dataloader = DataLoader(device_data, batch_size=batch_size, shuffle=True, drop_last=True)
        if vae_choice == 'mine':
            # define the stacked_bayesian_autoencoder object
            VAE = stacked_bayesian_autoencoder(original_dim=denoised_data.size()[1], im_dim=64, lat_dim=15)
            print(VAE)
            # Train the model
            encode.train_model(VAE, denoised_data, beta = 50, EPOCHS_0=encode_epochs[0], EPOCHS_1=encode_epochs[1], BATCH_SIZE=batch_size)
        elif vae_choice == "paper":
            # define the stacked_bayesian_autoencoder object
            VAE = paper_encoder(original_dim=denoised_data.size()[1], im_dim=64, lat_dim=15)
            print(VAE)
            # Train the model
            paper_encode.train_model(VAE, denoised_data, beta = 50, EPOCHS_0=encode_epochs[0], EPOCHS_1=encode_epochs[1], BATCH_SIZE=batch_size)
        
        # stop updating the model's parameters
        VAE.eval()
        
        # return the latent variable
        with torch.no_grad():
            latent = VAE.encode_mu(denoised_data).cpu().numpy()
            # print(latent.requires_grad)
            # torch.save(latent, f'latent/{path_name}.pt')
            # print(latent.cpu().numpy())
            # np.save(f'latent/{path_name}.npy', latent.cpu().numpy())
        print(latent, latent.max(), latent.min())
        return latent

