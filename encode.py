import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

import torch.optim as optim                          # optimization
from tqdm import tqdm                                # for progress bar
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
def sampling(mu:torch.Tensor, var:torch.Tensor, epsilon_std:torch.Tensor, lat_dim):
    return mu + torch.sqrt(var)*epsilon_std*torch.randn((1, lat_dim), device=device)

class paper_encoder(nn.Module):
    def __init__(self, original_dim, im_dim, lat_dim, epsilon_std=0.25, batch_norm=True, zero_bias=True):
        super().__init__()
        self.batch_norm = batch_norm
        self.epsilon_std = epsilon_std
        self.lat_dim = lat_dim
        
            
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
        #  generate multiple realizations of the input. This step makes the VAE more robust.
        self.decoder = nn.ModuleList(
            [nn.Linear(im_dim, original_dim, device=device),  # 64 -> input (5000)
             nn.Linear(im_dim, original_dim, device=device)]  # 64 -> input (5000)
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
            
        return out
        
    def encode_mu(self, x):
        if self.batch_norm:
            im = self.bn(self.encoder(x))
            mu = self.mu(im)
        else:
            im = F.selu(self.encoder(x))
            mu = self.mu(im)
        
        return mu
    
    
def train_model(model, data, EPOCHS_0 = 10, EPOCHS_1 = 20, BATCH_SIZE = 100, beta = 50):
    
    print("training stacked bayesian autoencoder")
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-7, weight_decay=1e-6)
    
    # the paper does the training data by data, but I choose to train in batches
    
    # the warm-up process, which uses only reconstruction loss
    print("\n###############################\n#phase 1: the warm-up process#\n######################################")
    for epoch in range(EPOCHS_0):
        optimizer.zero_grad()
        for i in tqdm(range(0, len(data), BATCH_SIZE)):
            
            # Forward
            output = model(data[i:i+BATCH_SIZE])
            # print(output)
            # output = [mu, var, output_0, output_1]
            loss = F.l1_loss(output[2], data[i:i+BATCH_SIZE]) + F.l1_loss(output[3], data[i:i+BATCH_SIZE])
            
            # print(f'loss1: {F.l1_loss(output[2], data[i:i+BATCH_SIZE])}, loss2: {F.l1_loss(output[3], data[i:i+BATCH_SIZE])}')
            torch.clamp(loss, max=0.5)
            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        # Show progress
        if epoch%10 == 9:
            print(f"epoch {epoch} Loss: {loss.item()}")
        #print(f'Loss: {loss.item()}, isnan: {torch.isnan(model.parameters()).any()}')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-7, weight_decay=1e-3)
    # the VAE stage training
    print("\n##############################\n#phase 2: the VAE stage#\n############################")
    for epoch in range(EPOCHS_1):
        optimizer.zero_grad()
        for i in tqdm(range(0, len(data), BATCH_SIZE)):
            # Forward
            output = model(data[i:i+BATCH_SIZE])
            loss = encode.loss_function(output[2], data[i:i+BATCH_SIZE], mu=output[0], var=output[1], beta=beta)\
                   + encode.loss_function(output[3], data[i:i+BATCH_SIZE], mu=output[0], var=output[1], beta=beta)
            
            # print(loss, loss.shape)
            
                
            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        # Show progress
        if epoch%10 == 9:
            print(f"epoch {epoch} Loss: {loss.item()}")
        # print(f'Loss: {loss.item()}, isnan: {torch.isnan(model.parameters()).any()}')
                
                
    
    