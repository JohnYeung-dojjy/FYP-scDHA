import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from torch_util import check_grad_nan

import torch.optim as optim                          # optimization
from tqdm import tqdm                                # for progress bar

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def sampling(mu:torch.Tensor, var:torch.Tensor, epsilon_std:torch.Tensor, lat_dim):
    return mu + torch.sqrt(var)*epsilon_std*torch.randn((1, lat_dim), device=device)

class stacked_bayesian_autoencoder(nn.Module):
    # im = intermediate
    # lat = latent
    def __init__(self, original_dim, im_dim, lat_dim, epsilon_std=0.25, batch_norm=True, zero_bias=True):
        super().__init__()
        
        self.batch_norm = batch_norm
        self.epsilon_std = epsilon_std
        self.lat_dim = lat_dim
        
        self.layer_1_dim = math.floor(original_dim/2)
            
        if(batch_norm):
            self.encoder = nn.Linear(original_dim, im_dim, device=device)
            self.bn = nn.BatchNorm1d(im_dim, momentum = 0.01, eps = 1e-3, device=device)
        else:
            self.encoder = nn.Linear(original_dim, im_dim, device=device)   # input -> 64
            
        self.mu = nn.Linear(im_dim, lat_dim, device=device)  # 64 -> 15
        self.var = nn.Linear(im_dim, lat_dim, device=device) # 64 -> 15
        
        self.sample_expander = nn.ModuleList(
            [nn.Linear(lat_dim, im_dim, device=device),     # 15 -> 64
             nn.Linear(lat_dim, im_dim, device=device)]
        )
        self.decoder = nn.ModuleList(
            [nn.Linear(im_dim, self.layer_1_dim               , device=device),  # 64 -> input[0:2500]
             nn.Linear(im_dim, original_dim - self.layer_1_dim, device=device)]  # 64 -> input[2500:5000]
        )
        
        # ========== init values =============
        torch.nn.init.xavier_uniform_(self.encoder.weight)
        torch.nn.init.xavier_uniform_(self.mu.weight)
        torch.nn.init.xavier_uniform_(self.var.weight)
        
        torch.nn.init.xavier_uniform_(self.sample_expander[0].weight)
        torch.nn.init.xavier_uniform_(self.sample_expander[1].weight)
        
        torch.nn.init.xavier_uniform_(self.decoder[0].weight)
        torch.nn.init.xavier_uniform_(self.decoder[1].weight) 
        
        if zero_bias:
            torch.nn.init.zeros_(self.encoder.bias)
            torch.nn.init.zeros_(self.mu.bias)
            torch.nn.init.zeros_(self.var.bias)
            
            torch.nn.init.zeros_(self.sample_expander[0].bias)
            torch.nn.init.zeros_(self.sample_expander[1].bias)
            
            torch.nn.init.zeros_(self.decoder[0].bias)
            torch.nn.init.zeros_(self.decoder[1].bias)
            
        
            
    def forward(self, x):
        if self.batch_norm:
            im = self.bn(self.encoder(x))
            mu = self.mu(im)
            var = F.softmax(self.var(im), dim=1)
        else:
            im = F.selu(self.encoder(x))
            mu = self.mu(im)
            var = F.softmax(self.var(im), dim=1)
        
        out = [mu, var]
        for i in range(2):
            z = sampling(mu, var, self.epsilon_std, self.lat_dim)
            out.append(self.decoder[i]( F.selu( self.sample_expander[i](z))))
            
        # print(len(out), out)
        return out
        
    def encode_mu(self, x):
        if self.batch_norm:
            im = self.bn(self.encoder(x))
            mu = self.mu(im)
        else:
            im = F.selu(self.encoder(x))
            mu = self.mu(im)
        
        return mu
    
def kl_div_loss(mu, var):
    loss = -0.5*torch.mean(1 + torch.log(var) - torch.square(mu) - var, dim=1)
    return loss
    

    #@torch.jit.script
def loss_function(input, target, mu, var, beta = 100):
    mse_loss = F.mse_loss(input, target)
    # kl_div = F.kl_div(input, target, reduction='batchmean')
    kl_div = kl_div_loss(mu, var)
    # print("mse_loss: ", mse_loss.item(), '\n', "kl_div: ", kl_div.item())
    return beta * mse_loss + kl_div

def train_model(model, data, EPOCHS_0 = 10, EPOCHS_1 = 20, BATCH_SIZE = 100, beta = 50):
    
    print("training stacked bayesian autoencoder")
    model.train()
    
    optimizer_0 = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer_1 = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # the paper does the training data by data, but I choose to train in batches
    
    # the warm-up process, which uses only reconstruction loss
    print("phase 1: the warm-up process")
    for epoch in range(EPOCHS_0):
        
        optimizer_0.zero_grad()
        for i in tqdm(range(0, len(data), BATCH_SIZE)):
            # Forward
            output = model(data[i:i+BATCH_SIZE])
            
            # output = [mu, var, output_0, output_1]
            loss = F.l1_loss(output[2], data[i:i+BATCH_SIZE, :model.layer_1_dim])\
                   + F.l1_loss(output[3], data[i:i+BATCH_SIZE, model.layer_1_dim:])
            # backward
            loss.backward()
            #nn.utils.clip_grad_norm(model.parameters(), max_norm=0.001)
            if check_grad_nan(model.parameters()):
                optimizer_0.zero_grad()
                continue
            optimizer_0.step()
            # optimizer_0.zero_grad()
            
        # Show progress
        print("Loss: ", loss.item())
        # print(f'Loss: {loss.item()}, isnan: {torch.isnan(model.parameters()).any()}')

    # the VAE stage training
    print("phase 2: the VAE stage")
    for epoch in range(EPOCHS_1):
        optimizer_1.zero_grad()
        for i in tqdm(range(0, len(data), BATCH_SIZE)):
            # Forward
            output = model(data[i:i+BATCH_SIZE])
            loss = loss_function(output[2], data[i:i+BATCH_SIZE, :model.layer_1_dim], mu=output[0], var=output[1], beta=beta)\
                   + loss_function(output[3], data[i:i+BATCH_SIZE, model.layer_1_dim:], mu=output[0], var=output[1], beta=beta)
            
            
            # backward
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), max_norm=0.001)
            if check_grad_nan(model.parameters()):
                optimizer_1.zero_grad()
                continue
            optimizer_1.step()
            # optimizer_1.zero_grad()
            
        # Show progress
        print("Loss: ", loss.item())
        # print(f'Loss: {loss.item()}, isnan: {torch.isnan(model.parameters()).any()}')

                
                
    
    