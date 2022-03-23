import pandas as pd
import numpy as np

import torch
import random

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import os
from tqdm import tqdm                                # for progress bar
from sklearn.model_selection import train_test_split


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


def scDHA(path_name, is_plot_denoise, vae_choice, retrain=False, seed=None):
    """load the data and compress it using the scDHA pipeline

    Args:
        path_name (string): the path_name to the data
        is_plot_denoise (bool): if plot the denoised result or not
        vae_choice (string): scDHA or my modified version
        retrain (bool, optional): retrain the VAE and generate new latent. Defaults to False.
        seed ([int], optional): random number seed, so result can be reproduced. Defaults to None.

    Returns:
        list: a list of 9 versions of generated latent variables
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
        
        random.seed(seed)
        ######### training variables ###########
        wdecay = [1e-6, 1e-3]
        batch_size = max(round(len(data)/50), 2)
        denoise_epochs = 10
        encode_epochs = [10, 20]
        orginal_dim = data.shape[1]
        
        lr = 5e-4
        epsilon_std = 0.25
        beta = 100
        ens = 3
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
        # print(f"first column of normalized_data\n{normalized_data[:,0]}")
        
        device_data = torch.tensor(normalized_data, dtype=torch.float).to(device)

        # Train the model
        # denoise.train_model(XNeg_kernel_autoencoder, dataloader, EPOCHS=denoise_epochs, BATCH_SIZE=batch_size)
        
        print("training non-negative kernel autoencoder")
        # gene filtering
        if seed is not None:
            random.seed(seed)
        Wsds = [] # standard deviations of Weights
        for i in range(3): # repeat for 3 times and take the average variance of weights
            if seed is not None:
                random.seed(seed + i)
                torch.manual_seed(seed + i)
            dataloader = DataLoader(device_data, batch_size=batch_size, shuffle=True, drop_last=True)
        
            # define the non_negative_kerel_autoencoder object
            XNeg_kernel_autoencoder = non_negative_kernel_autoencoder(orginal_dim, 50)
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
                    print(f"epoch {epoch+1} Loss: ", loss.item())
        
            # calculate variance of each features
            # https://pytorch.org/docs/stable/generated/torch.var.html
            # https://pytorch.org/docs/stable/generated/torch.transpose.html
            # weight_var = torch.std(torch.transpose(self.encoder.weight, 0, 1), dim=1)
            Wsd = torch.std(XNeg_kernel_autoencoder.encoder.weight.data, dim=0)
            
            # rescale weight_var to [0, 1] for ploting use
            Wsd[torch.isnan(Wsd)] = 0 # I don't know why this is here
            Wsd = (Wsd - Wsd.min()) / (Wsd.max() - Wsd.min())
            Wsd[torch.isnan(Wsd)] = 0 # in case of 0 division
            
            Wsds.append(Wsd.cpu().detach()) # add to Wsds
        
        Wsds = torch.stack((Wsds[0], Wsds[1], Wsds[2]), 0).mean(dim=0) # mean of each row
        
        # stop updating the model's parameters
        XNeg_kernel_autoencoder.eval()
        
        # choose 5000 most important features
        denoised_data = denoise.K_MostImportant_features(normalized_data, Wsds, plot=is_plot_denoise)
        
        # latent generating, generate 9 versions of latent variables (9 VAE of different weights)
        latent_tmp = []
        for i in range(3):
            if seed is not None:
                random.seed(seed + i)
                torch.manual_seed(seed + i)
                
            dataloader = DataLoader(denoised_data, batch_size=batch_size, shuffle=True, drop_last=True)
            # define the stacked_bayesian_autoencoder object
            VAE = paper_encoder(original_dim=denoised_data.size()[1], im_dim=64, lat_dim=15).to(device)
            # print(VAE)
            # Train the model
            print("training stacked bayesian autoencoder")
            VAE.train()
            
            optimizer = torch.optim.AdamW(VAE.parameters(), lr=5e-4, eps=1e-7, weight_decay=wdecay[0])
            
            # the paper does the training data by data, but I choose to train in batches
            
            # the warm-up process, which uses only reconstruction loss
            print("\n##############################\n#phase 1: the warm-up process#\n##############################")
            for epoch in range(encode_epochs[0]):
                optimizer.zero_grad()
                for data in dataloader:
                    data = data.to(device)
                    # Forward
                    output = VAE(data)
                    # print(output)
                    # output = [mu, var, output_0, output_1]
                    loss = F.l1_loss(output[2], data) + F.l1_loss(output[3], data)
                    
                    # torch.clamp(loss, max=0.5)
                    # backward
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                # Show progress
                if epoch%10 == 9:
                    print(f"epoch {epoch+1} Loss: {loss.item()}")
                #print(f'Loss: {loss.item()}, isnan: {torch.isnan(model.parameters()).any()}')
            
            optimizer = torch.optim.AdamW(VAE.parameters(), lr=lr, eps=1e-7, weight_decay=wdecay[1])
            # the VAE stage training
            print("\n########################\n#phase 2: the VAE stage#\n########################")
            for epoch in range(encode_epochs[1]):
                optimizer.zero_grad()
                for data in dataloader:
                    data = data.to(device)
                    # Forward
                    output = VAE(data)
                    loss = encode.loss_function(output[2], data, mu=output[0], var=output[1], beta=beta)\
                        + encode.loss_function(output[3], data, mu=output[0], var=output[1], beta=beta)
                    
                    # print(loss, loss.shape)
                    
                        
                    # backward
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                # Show progress
                if epoch%10 == 9:
                    print(f"epoch {epoch+1} Loss: {loss.item()}")
                # print(f'Loss: {loss.item()}, isnan: {torch.isnan(model.parameters()).any()}')
            for ite in range(3):
                VAE.to(device)
                VAE.train()
                optimizer.zero_grad()
                for data in dataloader:
                    data = data.to(device)
                    output = VAE(data)
                    loss = encode.loss_function(output[2], data, mu=output[0], var=output[1], beta=beta)\
                        + encode.loss_function(output[3], data, mu=output[0], var=output[1], beta=beta)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                # stop updating the model's parameters
                VAE.eval()
                
                # return the latent variable
                with torch.no_grad():
                    VAE.to('cpu') # avoid putting too much data to GPU and cause memory issue
                    tmp = VAE.encode_mu(denoised_data).cpu().numpy()
                    # print(latent.requires_grad)
                    # torch.save(latent, f'latent/{path_name}.pt')
                    # print(latent.cpu().numpy())
                    
                # print(latent, latent.max(), latent.min())
                latent_tmp.append(tmp)
        latent_tmp = np.array(latent_tmp)
        np.save(f'latent/{path_name}.npy', latent_tmp)
        # latent_tmp is a list of 9
        return latent_tmp
