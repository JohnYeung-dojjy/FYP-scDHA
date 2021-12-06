import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim                          # optimization
from tqdm import tqdm                                # for progress bar

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def normalization(data):
    # non-negative normalization, all data are from 0-1
    data_min = data.min(axis=0)[0]
    data_max = data.max(axis=0)[0]
    
    max_minus_min = (data_max - data_min) 
    
    # avoid 0 division
    max_minus_min = 1 if max_minus_min == 0 else max_minus_min
    
    normalized_data = (data - data_min) / max_minus_min
    
    return normalized_data

class non_negative_kernel_autoencoder(nn.Module):
    def __init__(self, input_dim, BOTTLENECK_SIZE = 50):
        super().__init__()

        self.encoder = nn.Linear(input_dim, BOTTLENECK_SIZE, device=device) # "The size of the bottleneck layer is set to 50 nodes
        
        self.decoder = nn.Linear(BOTTLENECK_SIZE, input_dim, device=device)
        
        torch.nn.init.xavier_uniform_(self.encoder.weight)
        torch.nn.init.zeros_(self.encoder.bias)
        
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        torch.nn.init.zeros_(self.decoder.bias)
        
        
    def forward(self, x):
    #     print(x.get_device())
    #     print(
    #           self.encoder.weight.get_device(),
    #           self.encoder.bias.get_device(),
    #           self.data.get_device())
        # x = F.relu(self.encoder(x)))
        x = self.encoder(x)
        
        x = F.softmax(self.decoder(x), dim=0)
        #x = self.decoder(x)
        
        return x
    
    def K_MostImportant_features(self, data, k, plot=False):
        data = torch.tensor(data, dtype=torch.float).to(device)
        
        if k > data.shape[0]:
            k = data.shape[0]
        
        # calculate variance of each features
        # https://pytorch.org/docs/stable/generated/torch.var.html
        # https://pytorch.org/docs/stable/generated/torch.transpose.html
        weight_var = torch.var(torch.transpose(self.encoder.weight, 0, 1), dim=1)
        # print(weight_var.data)
        # rescale weight_var to [0, 1] for ploting use
        weight_var = normalization(weight_var)
        
        if plot:
            # print(weight_var)
            plt.bar(np.linspace(0, data.shape[1], data.shape[1]), weight_var.cpu().detach().numpy())
            plt.xlabel('Genes')
            plt.ylabel('Normalized Weight Variance')
            plt.show()
            
        # find k features with highest variance
        K_highest_weight_var = torch.topk(weight_var, k)
        
        # location the features with corresponding indices
        features = torch.index_select(data, dim=1, index=K_highest_weight_var.indices)
        
        return features # (n, 5000)

def train_model(model: non_negative_kernel_autoencoder, data, EPOCHS = 10, BATCH_SIZE = 100):
    print("training non-negative kernel autoencoder")
    model.train()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
#         BATCH_SIZE = 2000
#         EPOCHS = 50
    # Train
    data = torch.tensor(data, dtype=torch.float).to(device)
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        for i in tqdm(range(0, len(data), BATCH_SIZE)):
            # Forward
            decoded_data = model(data[i:i+BATCH_SIZE])

            # Backward
            loss = loss_function(decoded_data, data[i:i+BATCH_SIZE])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            
            with torch.no_grad():
                model.encoder.weight.clamp_(min = 0) # inplace clamp
            
            # model.encoder.weight = Parameter(model.encoder.weight.detach().clamp(min = 0))
            
        # Show progress
        print("Loss: ", loss.item())
        # print(f'Loss: {loss.item()}, encoder gradient: {model.encoder.weight.grad} decoder gradient: {model.decoder.weight.grad}')